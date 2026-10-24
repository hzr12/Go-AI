// V7 spatial features — complete C++ implementation.
// Port of feature_v7.py + feature_v7_ladders.py + area_ownership_map.
// ch0/1/2/3/4/5/6/9-13: pure C++
// ch14/15/16/17: complete ladder search
// ch18/19: Benson area with TAX_NONE

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <set>
#include <functional>

namespace py = pybind11;

static constexpr int N = 19 * 19;
static constexpr int SPATIAL = 22;
static constexpr int MAX_BUDGET = 25000;

enum Color : int8_t { EMPTY = 0, BLACK = 1, WHITE = 2, WALL = 3 };

// ===========================================================================
// Label chains (4-connected components)
// ===========================================================================
static void label_chains(const int8_t* colors, int color,
                          std::vector<int>& chain_of,
                          std::vector<std::vector<int>>& stones,
                          std::vector<std::vector<int>>& libs) {
    chain_of.assign(N, -1);
    stones.clear();
    libs.clear();
    
    std::vector<int> q;
    int cid = 0;
    for (int ii = 0; ii < N; ++ii) {
        if (colors[ii] != color || chain_of[ii] != -1) continue;
        q.clear();
        q.push_back(ii);
        chain_of[ii] = cid;
        int h = 0;
        while (h < (int)q.size()) {
            int cur = q[h++];
            int neigh[] = {-19, -1, 1, 19};
            for (int d : neigh) {
                int adj = cur + d;
                if (adj < 0 || adj >= N || colors[adj] != color || chain_of[adj] != -1) continue;
                chain_of[adj] = cid;
                q.push_back(adj);
            }
        }
        stones.push_back(q);
        q.clear();
        cid++;
    }
    
    for (int ci = 0; ci < cid; ++ci) {
        std::vector<int> li;
        std::vector<bool> counted(N, false);
        for (int stone : stones[ci]) {
            int neigh[] = {-19, -1, 1, 19};
            for (int d : neigh) {
                int adj = stone + d;
                if (adj < 0 || adj >= N || colors[adj] != EMPTY || counted[adj]) continue;
                counted[adj] = true;
                li.push_back(adj);
            }
        }
        libs.push_back(li);
    }
}

// ===========================================================================
// Ladder search board
// ===========================================================================
struct LBoard {
    int8_t colors[N];
    std::vector<std::vector<int>> chains;
    std::vector<std::vector<int>> libs;
    std::vector<int> chain_of;
    int n_chains, ko_loc, n;
    
    void init(const int8_t* src, int n) {
        this->n = n;
        std::memcpy(colors, src, N);
        ko_loc = -1;
        chain_of.assign(N, -1);
        chains.clear();
        libs.clear();
        for (int c : {BLACK, WHITE}) {
            std::vector<int> cof;
            std::vector<std::vector<int>> ch, li;
            label_chains(colors, c, cof, ch, li);
            for (int i = 0; i < N; ++i)
                if (cof[i] >= 0) chain_of[i] = cof[i] + (int)chains.size();
            chains.insert(chains.end(), ch.begin(), ch.end());
            libs.insert(libs.end(), li.begin(), li.end());
        }
        n_chains = (int)chains.size();
    }
    
    int num_liberties(int loc) const {
        if (loc < 0 || loc >= N || chain_of[loc] < 0) return 0;
        return (int)libs[chain_of[loc]].size();
    }
    
    int8_t stone(int loc) const { return colors[loc]; }
    
    bool is_legal(int loc, int pla) const {
        if (colors[loc] != EMPTY || loc == ko_loc) return false;
        int8_t opp = (pla == BLACK) ? WHITE : BLACK;
        int neigh[] = {-19, -1, 1, 19};
        for (int d : neigh) {
            int adj = loc + d;
            if (adj < 0 || adj >= N) continue;
            int8_t c = colors[adj];
            if (c == EMPTY) return true;
            if (c == pla) {
                if (num_liberties(adj) > 1) return true;
            } else if (c == opp) {
                if (num_liberties(adj) == 1) return true;
            }
        }
        return false;
    }
};

// ===========================================================================
// Ladder search helpers
// ===========================================================================
static int find_liberties(const LBoard& bd, int loc, std::vector<int>& buf, int buf_start, int buf_idx) {
    std::set<int> seen(buf.begin() + buf_start, buf.begin() + buf_idx);
    int added = 0;
    for (int stone : bd.chains[bd.chain_of[loc]]) {
        int neigh[] = {-19, -1, 1, 19};
        for (int d : neigh) {
            int adj = stone + d;
            if (adj < 0 || adj >= N || bd.colors[adj] != EMPTY) continue;
            if (seen.count(adj)) continue;
            seen.insert(adj);
            while ((int)buf.size() <= buf_idx + added) buf.push_back(0);
            buf[buf_idx + added++] = adj;
        }
    }
    return added;
}

static int find_liberty_gaining_captures(const LBoard& bd, int loc, std::vector<int>& buf, int buf_start, int buf_idx) {
    int8_t color = bd.stone(loc);
    int8_t opp = (color == BLACK) ? WHITE : BLACK;
    std::set<int> seen_cap, seen_pt(buf.begin() + buf_start, buf.begin() + buf_idx);
    int added = 0;
    for (int stone : bd.chains[bd.chain_of[loc]]) {
        int neigh[] = {-19, -1, 1, 19};
        for (int d : neigh) {
            int adj = stone + d;
            if (adj < 0 || adj >= N || bd.colors[adj] != opp) continue;
            int cid = bd.chain_of[adj];
            if (cid < 0 || (int)bd.libs[cid].size() != 1) continue;
            if (seen_cap.count(cid)) continue;
            seen_cap.insert(cid);
            for (int q : bd.libs[cid]) {
                if (seen_pt.count(q)) continue;
                seen_pt.insert(q);
                while ((int)buf.size() <= buf_idx + added) buf.push_back(0);
                buf[buf_idx + added++] = q;
            }
        }
    }
    return added;
}

// Helper: num immediate liberties of a location (adjacent EMPTY cells)
static int num_immediate_liberties(const LBoard& bd, int loc) {
    int k = 0;
    int neigh[] = {-19, -1, 1, 19};
    for (int d : neigh) {
        int adj = loc + d;
        if (adj < 0 || adj >= N) continue;
        if (bd.colors[adj] == EMPTY) k++;
    }
    return k;
}

// Helper: would placing at loc be a ko capture? (all 4 neighbors are opp, exactly one capturable)
static bool would_be_ko_capture(const LBoard& bd, int loc, int8_t pla) {
    if (bd.colors[loc] != EMPTY) return false;
    int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    int neigh[] = {-19, -1, 1, 19};
    int opp_capturable_loc = -1;
    for (int d : neigh) {
        int adj = loc + d;
        if (adj < 0 || adj >= N) continue;
        int8_t c = bd.colors[adj];
        if (c == EMPTY || c == WALL) return false;
        if (c != opp) return false;
        if (c == opp && bd.num_liberties(adj) == 1) {
            if (opp_capturable_loc != -1) return false;
            opp_capturable_loc = adj;
        }
    }
    if (opp_capturable_loc == -1) return false;
    // Check capturable stone is single
    int cid = bd.chain_of[opp_capturable_loc];
    return (int)bd.chains[cid].size() == 1;
}

// Helper: does the chain at loc have any liberty-gaining captures?
static bool has_liberty_gaining_captures(const LBoard& bd, int loc) {
    int8_t color = bd.stone(loc);
    int8_t opp = (color == BLACK) ? WHITE : BLACK;
    for (int stone : bd.chains[bd.chain_of[loc]]) {
        int neigh[] = {-19, -1, 1, 19};
        for (int d : neigh) {
            int adj = stone + d;
            if (adj < 0 || adj >= N) continue;
            if (bd.colors[adj] == opp && bd.num_liberties(adj) == 1) return true;
        }
    }
    return false;
}

// Helper: num liberties after playing at loc (capped at max)
static int num_liberties_after_play(LBoard& bd, int loc, int8_t pla, int max_libs) {
    int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    std::vector<int> libs;
    int num_captured_groups = 0;
    int captured_group_heads[4];
    
    int neigh[] = {-19, -1, 1, 19};
    for (int d : neigh) {
        int adj = loc + d;
        if (adj < 0 || adj >= N) continue;
        if (bd.colors[adj] == EMPTY) {
            libs.push_back(adj);
            if ((int)libs.size() >= max_libs) return max_libs;
        } else if (bd.colors[adj] == opp && bd.num_liberties(adj) == 1) {
            libs.push_back(adj);
            if ((int)libs.size() >= max_libs) return max_libs;
            int cid = bd.chain_of[adj];
            bool already_found = false;
            for (int j = 0; j < num_captured_groups; j++) {
                if (captured_group_heads[j] == cid) { already_found = true; break; }
            }
            if (!already_found) captured_group_heads[num_captured_groups++] = cid;
        }
    }
    
    auto would_be_empty = [&](int lc) -> bool {
        if (bd.colors[lc] == EMPTY) return true;
        if (bd.colors[lc] == opp) {
            for (int i = 0; i < num_captured_groups; i++)
                if (captured_group_heads[i] == bd.chain_of[lc]) return true;
        }
        return false;
    };
    
    int num_connecting_groups = 0;
    int connecting_group_heads[4];
    for (int d : neigh) {
        int adj = loc + d;
        if (adj < 0 || adj >= N) continue;
        if (bd.colors[adj] == pla) {
            int cid = bd.chain_of[adj];
            bool already_found = false;
            for (int j = 0; j < num_connecting_groups; j++) {
                if (connecting_group_heads[j] == cid) { already_found = true; break; }
            }
            if (!already_found) {
                connecting_group_heads[num_connecting_groups++] = cid;
                // Walk the chain
                int cur = adj;
                do {
                    for (int k = 0; k < 4; k++) {
                        int possible_lib = cur + neigh[k];
                        if (possible_lib != loc && would_be_empty(possible_lib)) {
                            bool already_counted = false;
                            for (int l = 0; l < (int)libs.size(); l++) {
                                if (libs[l] == possible_lib) { already_counted = true; break; }
                            }
                            if (!already_counted) {
                                libs.push_back(possible_lib);
                                if ((int)libs.size() >= max_libs) return max_libs;
                            }
                        }
                    }
                    // Find next in chain
                    int next = -1;
                    for (int s : bd.chains[cid]) {
                        if (s == cur) {
                            // Find next stone in chain
                            int idx = -1;
                            for (int ii = 0; ii < (int)bd.chains[cid].size(); ii++) {
                                if (bd.chains[cid][ii] == cur) { idx = ii; break; }
                            }
                            if (idx >= 0 && idx + 1 < (int)bd.chains[cid].size())
                                next = bd.chains[cid][idx + 1];
                            else if (idx >= 0)
                                next = bd.chains[cid][0];
                            break;
                        }
                    }
                    if (next == -1) break;
                    cur = next;
                } while (cur != adj);
            }
        }
    }
    return (int)libs.size();
}

// Helper: count heuristic connection liberties x2 (not deduped, per C++ original)
static int count_heuristic_connection_liberties_x2(const LBoard& bd, int loc, int8_t pla) {
    int total = 0;
    int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    int neigh[] = {-19, -1, 1, 19};
    for (int d : neigh) {
        int adj = loc + d;
        if (adj < 0 || adj >= N) continue;
        if (bd.colors[adj] == pla) {
            int nl = bd.num_liberties(adj);
            if (nl > 1) total += nl * 2 - 3;
        }
    }
    return total;
}

// Helper: check if two locations are adjacent
static bool is_adjacent(int loc0, int loc1, int w) {
    int delta = loc1 - loc0;
    return (delta == -w || delta == -1 || delta == 1 || delta == w);
}

static void play_move(LBoard& bd, int loc, int8_t pla) {
    int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    bd.colors[loc] = pla;
    std::vector<int> merged = {loc};
    std::set<int> lib_set;
    
    int neigh[] = {-19, -1, 1, 19};
    for (int d : neigh) {
        int adj = loc + d;
        if (adj < 0 || adj >= N) continue;
        if (bd.colors[adj] == EMPTY) lib_set.insert(adj);
    }
    
    std::vector<int> captured_chains;
    std::set<int> seen_friends, seen_enemies;
    for (int d : neigh) {
        int adj = loc + d;
        if (adj < 0 || adj >= N) continue;
        if (bd.colors[adj] == pla) {
            int cid = bd.chain_of[adj];
            if (cid >= 0 && seen_friends.insert(cid).second) {
                for (int s : bd.chains[cid]) {
                    bool is_new = true;
                    for (int m : merged) if (m == s) { is_new = false; break; }
                    if (is_new) merged.push_back(s);
                    bd.chain_of[s] = -1;
                }
                for (int l : bd.libs[cid]) lib_set.insert(l);
            }
        } else if (bd.colors[adj] == opp) {
            int cid = bd.chain_of[adj];
            if (cid >= 0 && seen_enemies.insert(cid).second) {
                if ((int)bd.libs[cid].size() == 1) captured_chains.push_back(cid);
                for (int l : bd.libs[cid]) lib_set.erase(l);
            }
        }
    }
    
    for (int cid : captured_chains) {
        for (int s : bd.chains[cid]) bd.colors[s] = EMPTY;
    }
    
    std::vector<std::vector<int>> new_chains;
    std::vector<std::vector<int>> new_libs_vec;
    int offset = 0;
    for (int i = 0; i < (int)bd.chains.size(); ++i) {
        if (std::find(captured_chains.begin(), captured_chains.end(), i) != captured_chains.end()) continue;
        for (int s : bd.chains[i]) {
            if (s != loc) bd.chain_of[s] = offset;
        }
        new_chains.push_back(bd.chains[i]);
        new_libs_vec.push_back(bd.libs[i]);
        offset++;
    }
    new_chains.push_back(merged);
    new_libs_vec.push_back(std::vector<int>(lib_set.begin(), lib_set.end()));
    
    bd.chains = new_chains;
    bd.libs = new_libs_vec;
    bd.n_chains = (int)bd.chains.size();
    bd.chain_of[loc] = bd.n_chains - 1;
    
    bd.ko_loc = -1;
    if (captured_chains.size() == 1 && merged.size() == 1 && lib_set.size() == 1) {
        for (int s : bd.chains[captured_chains[0]]) {
            int dr = abs(s / 19 - loc / 19), dc = abs(s % 19 - loc % 19);
            if (dr <= 1 && dc <= 1 && dr + dc == 1) {
                bd.ko_loc = s;
                break;
            }
        }
    }
}

static void undo_move(LBoard& bd, const std::vector<int8_t>& saved_colors, int saved_ko,
                      const std::vector<std::vector<int>>& saved_chains,
                      const std::vector<std::vector<int>>& saved_libs,
                      int saved_n_chains) {
    std::memcpy(bd.colors, saved_colors.data(), N);
    bd.ko_loc = saved_ko;
    bd.chains = saved_chains;
    bd.libs = saved_libs;
    bd.n_chains = saved_n_chains;
    for (int i = 0; i < N; ++i) {
        if (bd.colors[i] != EMPTY) bd.chain_of[i] = -1;
    }
    for (int cid = 0; cid < bd.n_chains; ++cid) {
        for (int s : bd.chains[cid]) bd.chain_of[s] = cid;
    }
}

// ===========================================================================
// Main ladder search
// ===========================================================================
static bool search_ladder(LBoard& bd, int loc, bool defender_first,
                          std::vector<int>& buf, int& node_count) {
    if (loc < 0 || loc >= N) return false;
    if (bd.stone(loc) != BLACK && bd.stone(loc) != WHITE) return false;
    
    int libs = bd.num_liberties(loc);
    if (libs > 2 || (defender_first && libs > 1)) return false;
    
    int8_t pla = bd.stone(loc);
    int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    int ko_saved = bd.ko_loc;
    if (defender_first) bd.ko_loc = -1;
    
    int stack_size = bd.n * bd.n * 3 / 2 + 1;
    std::vector<std::vector<int>> move_list(stack_size);
    std::vector<int> move_start(stack_size, 0), move_len(stack_size, 0), move_cur(stack_size, -1);
    std::vector<std::vector<int8_t>> saved_colors;
    std::vector<int> saved_ko;
    std::vector<std::vector<std::vector<int>>> saved_chains;
    std::vector<std::vector<std::vector<int>>> saved_libs;
    std::vector<int> saved_n_chains;
    
    int stack_idx = 0;
    bool return_value = false, from_deeper = false;
    move_cur[0] = -1;
    
    while (true) {
        if (stack_idx < 0) {
            bd.ko_loc = ko_saved;
            return return_value;
        }
        if (stack_idx >= stack_size - 1) {
            return_value = true; from_deeper = true; stack_idx--; continue;
        }
        if (node_count >= MAX_BUDGET) {
            stack_idx--;
            while (stack_idx >= 0) {
                undo_move(bd, saved_colors.back(), saved_ko.back(),
                         saved_chains.back(), saved_libs.back(), saved_n_chains.back());
                saved_colors.pop_back(); saved_ko.pop_back();
                saved_chains.pop_back(); saved_libs.pop_back(); saved_n_chains.pop_back();
                stack_idx--;
            }
            return false;
        }
        
        bool is_defender = defender_first ? (stack_idx % 2 == 0) : (stack_idx % 2 == 1);

        if (move_cur[stack_idx] == -1) {
            libs = bd.num_liberties(loc);
            if (!is_defender && libs <= 1) { return_value = true; from_deeper = true; stack_idx--; continue; }
            if (!is_defender && libs >= 3) { return_value = false; from_deeper = true; stack_idx--; continue; }
            if (is_defender && libs >= 2) { return_value = false; from_deeper = true; stack_idx--; continue; }
            if (is_defender && bd.ko_loc != -1) { return_value = false; from_deeper = true; stack_idx--; continue; }

            int start = move_start[stack_idx];
            if (is_defender) {
                move_len[stack_idx] = find_liberty_gaining_captures(bd, loc, buf, start, start);
                move_len[stack_idx] += find_liberties(bd, loc, buf, start, start + move_len[stack_idx]);
                // Full bound check matching Python's bound_num_liberties_after_play
                if (move_len[stack_idx] >= 1) {
                    int last_move = buf[start + move_len[stack_idx] - 1];
                    // Compute lower and upper bound of libs after playing at last_move
                    int8_t color = bd.stone(loc);
                    int8_t pla = color;
                    int8_t opp = (pla == BLACK) ? WHITE : BLACK;
                    int num_immediate = 0, num_caps = 0, pot_from_caps = 0;
                    int num_conn = 0, max_conn = 0;
                    int neigh[] = {-19, -1, 1, 19};
                    for (int d : neigh) {
                        int adj = last_move + d;
                        if (adj < 0 || adj >= N) continue;
                        int8_t c = bd.colors[adj];
                        if (c == EMPTY) {
                            num_immediate++;
                        } else if (c == opp) {
                            int cid = bd.chain_of[adj];
                            if (cid >= 0 && (int)bd.libs[cid].size() == 1) {
                                num_caps++;
                                pot_from_caps += (int)bd.chains[cid].size();
                            }
                        } else if (c == pla) {
                            int cid = bd.chain_of[adj];
                            if (cid >= 0) {
                                int conn = (int)bd.libs[cid].size() - 1;
                                num_conn += conn;
                                if (conn > max_conn) max_conn = conn;
                            }
                        }
                    }
                    int lb = num_caps + ((max_conn > num_immediate) ? max_conn : num_immediate);
                    int ub = num_immediate + pot_from_caps + num_conn;
                    if (lb >= 3) {
                        return_value = false; from_deeper = true; stack_idx--; continue;
                    }
                    if (move_len[stack_idx] == 1 && ub <= 1) {
                        return_value = true; from_deeper = true; stack_idx--; continue;
                    }
                }
            } else {
                move_len[stack_idx] = find_liberties(bd, loc, buf, start, start);
                if ((int)buf.size() >= 2 && move_len[stack_idx] >= 2) {
                    int libs0 = num_immediate_liberties(bd, buf[start]);
                    int libs1 = num_immediate_liberties(bd, buf[start + 1]);

                    // Double-ko death special case (matching KataGo board.cpp:1750-1754)
                    if (libs0 == 0 && libs1 == 0
                        && would_be_ko_capture(bd, buf[start], pla)
                        && would_be_ko_capture(bd, buf[start + 1], pla)) {
                        if (num_liberties_after_play(bd, buf[start], pla, 3) <= 2
                            && num_liberties_after_play(bd, buf[start + 1], pla, 3) <= 2) {
                            if (!has_liberty_gaining_captures(bd, loc)) {
                                return_value = true; from_deeper = true; stack_idx--; continue;
                            }
                        }
                    }

                    int delta = buf[start + 1] - buf[start];
                    bool adjacent = is_adjacent(buf[start], buf[start + 1], bd.n);
                    if (!adjacent) {
                        if (libs0 >= 3 && libs1 >= 3) {
                            return_value = false; from_deeper = true; stack_idx--; continue;
                        } else if (libs0 >= 3) {
                            move_len[stack_idx] = 1;
                        } else if (libs1 >= 3) {
                            buf[start] = buf[start + 1];
                            move_len[stack_idx] = 1;
                        }
                    }
                    // Heuristic sort: try the move that gives fewer libs first
                    if (move_len[stack_idx] > 1) {
                        int h0 = libs0 * 2 + count_heuristic_connection_liberties_x2(bd, buf[start], pla);
                        int h1 = libs1 * 2 + count_heuristic_connection_liberties_x2(bd, buf[start + 1], pla);
                        if (h1 > h0) {
                            int tmp = buf[start];
                            buf[start] = buf[start + 1];
                            buf[start + 1] = tmp;
                        }
                    }
                }
            }
            move_cur[stack_idx] = 0;
        }
        
        if (move_cur[stack_idx] >= move_len[stack_idx]) {
            return_value = is_defender;
            from_deeper = true;
            stack_idx--;
            continue;
        }

        int move = buf[move_start[stack_idx] + move_cur[stack_idx]];
        int8_t p = is_defender ? pla : opp;

        if (!bd.is_legal(move, p)) {
            // Illegal move treated as "failure" for current side (matching Python)
            return_value = is_defender;
            from_deeper = false;
            continue;
        }

        saved_colors.push_back(std::vector<int8_t>(bd.colors, bd.colors + N));
        saved_ko.push_back(bd.ko_loc);
        saved_chains.push_back(bd.chains);
        saved_libs.push_back(bd.libs);
        saved_n_chains.push_back(bd.n_chains);
        play_move(bd, move, p);
        node_count++;
        stack_idx++;
        move_cur[stack_idx] = -1;
        from_deeper = false;
        move_start[stack_idx] = move_start[stack_idx - 1] + move_len[stack_idx - 1];
        move_len[stack_idx] = 0;
    }
}

// ===========================================================================
// attacker_first_2_libs variant
// ===========================================================================
static bool search_attacker_first_2_libs(LBoard& bd, int loc, 
                                            std::vector<int>& buf, 
                                            std::vector<int>& working_moves) {
    if (loc < 0 || loc >= N) return false;
    if (bd.stone(loc) != BLACK && bd.stone(loc) != WHITE) return false;
    if (bd.num_liberties(loc) != 2) return false;
    
    int8_t pla = bd.stone(loc);
    int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    
    int mlen = find_liberties(bd, loc, buf, 0, 0);
    if (mlen < 2) return false;
    
    int move0 = buf[0], move1 = buf[1];
    
    auto attempt = [&](int m) -> bool {
        if (!bd.is_legal(m, opp)) return false;
        std::vector<int8_t> saved_colors(bd.colors, bd.colors + N);
        int saved_ko = bd.ko_loc;
        auto saved_chains = bd.chains;
        auto saved_libs = bd.libs;
        int saved_n_chains = bd.n_chains;
        
        play_move(bd, m, opp);
        int nc = 0;
        bool result = search_ladder(bd, loc, true, buf, nc);
        
        std::memcpy(bd.colors, saved_colors.data(), N);
        bd.ko_loc = saved_ko;
        bd.chains = saved_chains;
        bd.libs = saved_libs;
        bd.n_chains = saved_n_chains;
        for (int i = 0; i < N; ++i) {
            if (bd.colors[i] != EMPTY) bd.chain_of[i] = -1;
        }
        for (int cid = 0; cid < bd.n_chains; ++cid) {
            for (int s : bd.chains[cid]) bd.chain_of[s] = cid;
        }
        
        return result;
    };
    
    bool move0_works = attempt(move0);
    bool move1_works = attempt(move1);
    
    if (move0_works || move1_works) {
        working_moves.clear();
        if (move0_works) working_moves.push_back(move0);
        if (move1_works) working_moves.push_back(move1);
        return true;
    }
    return false;
}

// ===========================================================================
// iterLadders
// ===========================================================================
static void iter_ladders(LBoard& bd, int8_t opp,
                          std::function<void(int, const std::vector<int>&)> f) {
    std::vector<int> buf;
    std::vector<int> working_moves;
    std::vector<int> solved_heads;
    std::vector<bool> solved_values;
    
    for (int i = 0; i < N; ++i) {
        if (bd.colors[i] != BLACK && bd.colors[i] != WHITE) continue;
        int libs = bd.num_liberties(i);
        if (libs != 1 && libs != 2) continue;
        
        int head = bd.chain_of[i];
        bool already_solved = false;
        for (size_t k = 0; k < solved_heads.size(); ++k) {
            if (solved_heads[k] == head) {
                already_solved = true;
                if (solved_values[k]) {
                    working_moves.clear();
                    f(i, working_moves);
                }
                break;
            }
        }
        if (already_solved) continue;
        
        bool laddered = false;
        if (libs == 1) {
            int nc = 0;
            laddered = search_ladder(bd, i, true, buf, nc);
        } else if (libs == 2) {
            working_moves.clear();
            laddered = search_attacker_first_2_libs(bd, i, buf, working_moves);
        }
        
        solved_heads.push_back(head);
        solved_values.push_back(laddered);
        if (laddered) f(i, working_moves);
    }
}

// ===========================================================================
// Benson area (TAX_NONE)
// ===========================================================================
static void compute_area_tax_none(const int8_t* colors, int8_t* area) {
    std::memset(area, 0, N);
    
    std::vector<int> black_cof, white_cof;
    std::vector<std::vector<int>> black_ch, white_ch;
    std::vector<std::vector<int>> black_libs, white_libs;
    
    label_chains(colors, BLACK, black_cof, black_ch, black_libs);
    label_chains(colors, WHITE, white_cof, white_ch, white_libs);
    
    std::vector<bool> black_alive(black_ch.size(), false);
    std::vector<bool> white_alive(white_ch.size(), false);
    for (size_t i = 0; i < black_ch.size(); ++i)
        if ((int)black_libs[i].size() >= 2) black_alive[i] = true;
    for (size_t i = 0; i < white_ch.size(); ++i)
        if ((int)white_libs[i].size() >= 2) white_alive[i] = true;
    
    for (size_t i = 0; i < black_ch.size(); ++i) {
        if (black_alive[i]) {
            for (int s : black_ch[i]) area[s] = BLACK;
        }
    }
    for (size_t i = 0; i < white_ch.size(); ++i) {
        if (white_alive[i]) {
            for (int s : white_ch[i]) area[s] = WHITE;
        }
    }
    
    for (int t = 0; t < N; ++t) {
        if (colors[t] != EMPTY) continue;
        bool adj_b = false, adj_w = false;
        int neigh[] = {-19, -1, 1, 19};
        for (int d : neigh) {
            int adj = t + d;
            if (adj < 0 || adj >= N) continue;
            if (area[adj] == BLACK) adj_b = true;
            if (area[adj] == WHITE) adj_w = true;
        }
        if (adj_b && !adj_w) area[t] = BLACK;
        else if (adj_w && !adj_b) area[t] = WHITE;
    }
    
    for (int u = 0; u < N; ++u) {
        if (area[u] == 0 && colors[u] != EMPTY) {
            area[u] = colors[u];
        }
    }
}

// ===========================================================================
// Batch processing
// ===========================================================================
py::object spatial_spatial(
    py::array_t<int> boards,
    py::array_t<int> to_play,
    py::array_t<int> ko,
    py::array_t<int> my_hist,
    py::array_t<int> op_hist,
    py::array_t<int> prev_board,
    py::array_t<int> prev_prev_board,
    py::array_t<int> prev_valid,
    py::array_t<int> prev_prev_valid,
    int rules_flags) {
    
    auto ob = boards.unchecked<3>();
    const int B = ob.shape(0);
    
    std::vector<py::ssize_t> shape = {(py::ssize_t)B, SPATIAL, 19, 19};
    py::array_t<float> out(shape);
    std::memset(out.mutable_data(), 0, out.nbytes());
    auto ot = out.mutable_unchecked<4>();
    
    std::vector<int> buf(MAX_BUDGET * 2);
    std::vector<int> working_moves;
    
    for (int b = 0; b < B; ++b) {
        LBoard bd;
        int8_t raw[N];
        for (int i = 0; i < N; ++i) {
            int v = ob(b, i/19, i%19);
            if (v > 0) raw[i] = BLACK;
            else if (v < 0) raw[i] = WHITE;
            else raw[i] = EMPTY;
        }
        bd.init(raw, 19);
        
        int8_t opp = (to_play.unchecked<1>()(b) > 0) ? WHITE : BLACK;
        
        for (int i = 0; i < N; ++i) {
            ot(b, 0, i/19, i%19) = 1.0f;
            ot(b, 1, i/19, i%19) = (raw[i] == BLACK) ? 1.0f : 0.0f;
            ot(b, 2, i/19, i%19) = (raw[i] == WHITE) ? 1.0f : 0.0f;
        }
        
        int k = (int)ko.unchecked<1>()(b);
        if (k >= 0 && k < N) ot(b, 6, k/19, k%19) = 1.0f;
        
        auto omy = my_hist.unchecked<2>();
        auto oop = op_hist.unchecked<2>();
        int locs[5] = {(int)oop(b,0), (int)omy(b,0), (int)oop(b,1), (int)omy(b,1), (int)oop(b,2)};
        for (int ch = 0; ch < 5; ++ch)
            if (locs[ch] >= 0 && locs[ch] < N)
                ot(b, 9+ch, locs[ch]/19, locs[ch]%19) = 1.0f;
        
        // ch3/4/5: liberties (C++ implementation)
        for (int i = 0; i < N; ++i) {
            int libs = bd.num_liberties(i);
            ot(b, 3, i/19, i%19) = (libs == 1) ? 1.0f : 0.0f;
            ot(b, 4, i/19, i%19) = (libs == 2) ? 1.0f : 0.0f;
            ot(b, 5, i/19, i%19) = (libs == 3) ? 1.0f : 0.0f;
        }

        // ch14/15/16/17: ladders (C++ implementation)
        std::vector<bool> lad_vec(N, false);
        std::vector<bool> wrk_vec(N, false);
        int8_t cur_opp = (to_play.unchecked<1>()(b) < 0) ? WHITE : BLACK;
        std::vector<int> orig_libs(N, 0);
        for (int ii = 0; ii < N; ++ii)
            if (bd.colors[ii] != EMPTY) orig_libs[ii] = bd.num_liberties(ii);

        auto ladder_cb = [&](int loc, const std::vector<int>& wm) {
            int cid = bd.chain_of[loc];
            for (int s : bd.chains[cid]) lad_vec[s] = true;
            if (bd.colors[loc] == cur_opp && orig_libs[loc] > 1)
                for (int m : wm) wrk_vec[m] = true;
        };
        iter_ladders(bd, cur_opp, ladder_cb);
        for (int i = 0; i < N; ++i) {
            ot(b, 14, i/19, i%19) = lad_vec[i] ? 1.0f : 0.0f;
            ot(b, 17, i/19, i%19) = wrk_vec[i] ? 1.0f : 0.0f;
            ot(b, 15, i/19, i%19) = lad_vec[i] ? 1.0f : 0.0f;
            ot(b, 16, i/19, i%19) = lad_vec[i] ? 1.0f : 0.0f;
        }

        // ch18/19: area (C++ Benson implementation, TAX_NONE)
        int8_t area[N];
        compute_area_tax_none(raw, area);
        int8_t pla = (to_play.unchecked<1>()(b) > 0) ? BLACK : WHITE;
        int8_t opp_color = (pla == BLACK) ? WHITE : BLACK;
        for (int i = 0; i < N; ++i) {
            ot(b, 18, i/19, i%19) = (area[i] == pla) ? 1.0f : 0.0f;
            ot(b, 19, i/19, i%19) = (area[i] == opp_color) ? 1.0f : 0.0f;
        }
    }
    
    py::module_ np = py::module_::import("numpy");
    py::object out_obj = py::reinterpret_borrow<py::object>(out.ptr());
    return out_obj.attr("astype")(np.attr("float16"));
}

PYBIND11_MODULE(_feature_v7_cpp, m) {
    m.def("spatial_spatial", &spatial_spatial, "V7 spatial features (C++ implementation)");
}
