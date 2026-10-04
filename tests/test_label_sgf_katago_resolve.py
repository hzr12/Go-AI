"""AppImage 解析 + 依赖预检的回归测试。

2026-10-02 交付的 soft_tag.zip 用的引擎就是 AppImage，而目标环境（容器）没有
/dev/fuse，`./katago` 直接死在 FUSE 挂载上。症状还特别坑：报错不在引擎启动处，
而是绕到 `wait_ready()` 变成「240s 未就绪」，看起来像超时不像分发形态问题。

同时盯住一个**假警报**：`ldd` 裸跑看不到 AppRun 设的 LD_LIBRARY_PATH，会把
AppImage **自带的** libzip 报成「not found」。不修正就会把用户引去装一个他其实
已经有的库。
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util as _ilu  # noqa: E402

_s = _ilu.spec_from_file_location(
    "label_sgf",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "label_sgf.py"))
L = _ilu.module_from_spec(_s)
_s.loader.exec_module(L)


_ELF_HEAD = b"\x7fELF\x02\x01\x01\x00"


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def _fake_appimage(path):
    return _write(path, _ELF_HEAD + b"AI\x02" + b"\x00" * 100)


def _fake_plain_elf(path):
    return _write(path, _ELF_HEAD + b"\x02\x00\x3e\x00" + b"\x00" * 100)


# --------------------------------------------------------------------------- #
# is_appimage
# --------------------------------------------------------------------------- #
def test_is_appimage_detects_type2(tmp_path):
    assert L.is_appimage(_fake_appimage(tmp_path / "katago")) is True


def test_is_appimage_rejects_plain_elf(tmp_path):
    """ 关键：AppImage 本身就是 ELF。只查 ELF 魔数会把两者当成同一个东西 ——
    这正是当初只校验「是 ELF」就发包、结果在目标环境跑不起来的原因。"""
    assert L.is_appimage(_fake_plain_elf(tmp_path / "katago")) is False


def test_is_appimage_rejects_non_elf_and_missing(tmp_path):
    assert L.is_appimage(_write(tmp_path / "x", b"MZ\x90\x00")) is False
    assert L.is_appimage(str(tmp_path / "nope")) is False


def test_is_appimage_rejects_windows_exe(tmp_path):
    assert L.is_appimage(_write(tmp_path / "katago.exe", b"MZ" + b"\x00" * 20)) is False


# --------------------------------------------------------------------------- #
# resolve_katago
# --------------------------------------------------------------------------- #
def test_resolve_passes_through_plain_binary(tmp_path):
    """裸二进制原样返回，且不产生任何 AppImage 副作用。"""
    exe = _fake_plain_elf(tmp_path / "katago")
    got, libs = L.resolve_katago(exe)
    assert got == exe
    assert libs == ()
    assert not (tmp_path / "squashfs-root").exists()


def test_resolve_uses_apprun_when_already_extracted(tmp_path):
    """已解包 -> 走 AppRun（它会设好 LD_LIBRARY_PATH），且**不再解包**。"""
    exe = _fake_appimage(tmp_path / "katago")
    (tmp_path / "squashfs-root").mkdir()
    (tmp_path / "squashfs-root" / "AppRun").write_text("#!/bin/sh\n")
    got, libs = L.resolve_katago(exe)
    assert os.path.basename(got) == "AppRun"
    # 没建 lib 目录 -> libs 为空元组，仍然合法
    assert isinstance(libs, tuple)


def test_resolve_collects_bundled_libdirs(tmp_path):
    """ libzip 假警报的修复点：捆绑 so 的目录必须被收集出来交给 ldd。"""
    exe = _fake_appimage(tmp_path / "katago")
    root = tmp_path / "squashfs-root"
    (root / "usr" / "lib").mkdir(parents=True)
    (root / "usr" / "lib" / "libzip.so.4").write_bytes(b"x")
    (root / "AppRun").write_text("#!/bin/sh\n")
    _got, libs = L.resolve_katago(exe)
    assert any(d.endswith(os.path.join("usr", "lib")) for d in libs), libs


def test_resolve_raises_with_actionable_hint_when_not_extracted(tmp_path):
    """未解包 -> **不自动解包**，但错误必须给出可直接复制的解包命令。"""
    exe = _fake_appimage(tmp_path / "katago")
    with pytest.raises(SystemExit) as e:
        L.resolve_katago(exe)
    msg = str(e.value)
    assert "AppImage" in msg
    assert "--appimage-extract" in msg
    # 不得擅自落 ~100MB 磁盘副作用
    assert not (tmp_path / "squashfs-root").exists()


def test_resolve_handles_none_and_missing():
    assert L.resolve_katago(None) == (None, ())


# --------------------------------------------------------------------------- #
# missing_shared_libs
# --------------------------------------------------------------------------- #
def test_missing_libs_is_none_on_non_linux():
    """非 Linux 返回 None 而**不是空 list** —— 空 list 会被 main() 当成
    「检查通过」，在 Windows 冒烟时给出假的安心。"""
    if sys.platform.startswith("linux"):
        pytest.skip("本机是 Linux")
    assert L.missing_shared_libs("/bin/ls") is None


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="需要 ldd")
def test_missing_libs_parses_not_found(monkeypatch):
    out = ("libzip.so.4 => not found\n"
           "libcudnn.so.9 => not found\n"
           "libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00)\n")
    monkeypatch.setattr(L.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": out})())
    got = L.missing_shared_libs("/whatever")
    assert got == ["libcudnn.so.9", "libzip.so.4"]   # 已排序
    # 有条目的都在提示表里能查到处置办法
    for m in got:
        assert m in L._DEP_HINT


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="需要 ldd")
def test_missing_libs_passes_appimage_libdirs_to_ldd(monkeypatch):
    """ 假警报修复：必须把 libdirs 拼进 LD_LIBRARY_PATH 再跑 ldd，
    否则 AppImage 自带的 libzip 会被报成缺失。"""
    seen = {}

    def fake_run(cmd, **k):
        seen["env"] = k.get("env", {})
        return type("R", (), {"stdout": "libc.so.6 => /lib/libc.so.6\n"})()

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    L.missing_shared_libs("/x/AppRun", ("/x/usr/lib",))
    ldp = seen["env"]["LD_LIBRARY_PATH"]
    assert ldp.split(os.pathsep)[0] == "/x/usr/lib"


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="需要 ldd")
def test_missing_libs_does_not_clobber_existing_ldpath(monkeypatch):
    seen = {}
    monkeypatch.setattr(L.os.environ, "get", lambda k, d="": "/pre/existing")
    monkeypatch.setattr(L.subprocess, "run",
                        lambda *a, **k: (seen.update(env=k.get("env", {})),
                                         type("R", (), {"stdout": ""}))[1])
    L.missing_shared_libs("/x/AppRun", ("/x/usr/lib",))
    parts = seen["env"]["LD_LIBRARY_PATH"].split(os.pathsep)
    assert parts == ["/x/usr/lib", "/pre/existing"]   # 原有值在后，没被覆盖


def test_missing_libs_none_when_no_ldd(monkeypatch):
    """没有 ldd 的系统（如精简镜像）必须返回 None，不能抛。"""
    monkeypatch.setattr(L.sys, "platform", "linux")
    monkeypatch.setattr(L.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert L.missing_shared_libs("/x") is None
