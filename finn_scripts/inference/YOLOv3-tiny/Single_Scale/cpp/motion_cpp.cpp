/**
 * C++ motion detection
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace py = pybind11;

// Constants
static const int MOTION_CNT_ADD = 10;
static const int MOTION_CNT_DROP = 4;
static const int MOTION_CNT_LIMIT = 25;
static const int MIN_FILL_RATIO = 10;
static const int MIN_FILL_RATIO_IMMEDIATE = 20;
static const int MIN_FILL_RATIO_FAST = 50;
static const int FRAMES_FROM_MOTION_ADD = 100;
static const int FRAMES_FROM_MOTION_LIMIT = 400;
static const int FRAMES_FROM_MOTION_ADD_FAST = 60;
static const int FRAMES_FROM_MOTION_LIMIT_FAST = 60;
static const int FRAME_CNT_MOTION_VALIDATED = 4;
static const int FRAME_CNT_MOTION_NOT_VALIDATED = 8;
static const int MOTION_VALIDATION_FPS_MS_LIMIT = 400;

struct MotionWindowData {
    int x, y, width, height;
    int rough_area;
    int motion_cnt[4];
    int frames_from_motion;
    int frames_from_motion_fast;
    int motion_state;
    int frames_by_immediate_tests;
    int frames_by_immediate;
    int motion_super_fast;
    bool result_immediate_allowed;
    bool activate_off_by_motion;
    int xleft, ytop, xright, ybottom;
    int xmidleft, ymidtop, xmidright, ymidbottom;
};

class MotionEngine {
public:
    MotionEngine(int frame_width, int frame_height,
                 int cell_width, int cell_height, int noof_zooms)
        : half_w_(frame_width / 2)
        , half_h_(frame_height / 2)
    {
        create_windows(cell_width, cell_height, noof_zooms);
        diff_.resize(static_cast<size_t>(half_w_) * half_h_);
        sat_.resize(static_cast<size_t>(half_w_ + 1) * (half_h_ + 1));
    }

    int get_window_count() const { return static_cast<int>(windows_.size()); }

    int process_motion(py::array_t<uint8_t> prev_arr, py::array_t<uint8_t> curr_arr,
                       int frame_time_ms, bool can_process, bool do_reset) {
        py::buffer_info pb = prev_arr.request();
        py::buffer_info cb = curr_arr.request();
        int h = static_cast<int>(pb.shape[0]);
        int w = static_cast<int>(pb.shape[1]);
        int prev_ch = (pb.ndim >= 3) ? static_cast<int>(pb.shape[2]) : 1;
        int curr_ch = (cb.ndim >= 3) ? static_cast<int>(cb.shape[2]) : 1;
        const uint8_t* prev = static_cast<const uint8_t*>(pb.ptr);
        const uint8_t* curr = static_cast<const uint8_t*>(cb.ptr);
        size_t prev_stride = (pb.ndim >= 3) ? static_cast<size_t>(pb.strides[0]) : static_cast<size_t>(w);
        size_t curr_stride = (cb.ndim >= 3) ? static_cast<size_t>(cb.strides[0]) : static_cast<size_t>(w);

        build_diff_and_sat(prev, curr, w, h, prev_ch, curr_ch, prev_stride, curr_stride);
        return update_windows(frame_time_ms, can_process, do_reset);
    }

    py::tuple check_triggered(int x, int y, int width, int height) {
        int target_cx = x + width / 2;
        int target_cy = y + height / 2;
        int target_diff = width * width * 2 / 7 + height * height * 2 / 7;
        for (const auto& wnd : windows_) {
            int center_x = wnd.x + wnd.width / 2;
            int center_y = wnd.y + wnd.height / 2;
            int dx = target_cx - center_x;
            int dy = target_cy - center_y;
            if (dx * dx + dy * dy <= target_diff && wnd.activate_off_by_motion) {
                return py::make_tuple(true, wnd.motion_super_fast);
            }
        }
        return py::make_tuple(false, 0);
    }

private:
    int half_w_, half_h_;
    std::vector<MotionWindowData> windows_;
    std::vector<uint8_t> diff_;
    std::vector<int64_t> sat_;

    void create_windows(int cell_width, int cell_height, int noof_zooms) {
        int size_x = cell_width;
        int size_y = cell_height;
        double scale = 1.0;
        double scale_for_area = 1.0;

        for (int zoom = 0; zoom < noof_zooms; ++zoom) {
            int step_x = std::max(1, size_x / 2);
            int step_y = std::max(1, size_y / 2);
            int noof_half_x = half_w_ / step_x;
            int noof_half_y = half_h_ / step_y;
            if (noof_half_x < 2 || noof_half_y < 2) break;

            int y = (half_h_ - (noof_half_y - 1) * step_y) / 2;
            for (int idxy = 0; idxy < noof_half_y - 1; ++idxy) {
                int x = (half_w_ - (noof_half_x - 1) * step_x) / 2;
                for (int idxx = 0; idxx < noof_half_x - 1; ++idxx) {
                    int xleft = std::max(0, std::min(x, half_w_ - 1));
                    int ytop = std::max(0, std::min(y, half_h_ - 1));
                    int xright = std::min(xleft + std::max(1, size_x / 2), half_w_ - 1);
                    int ybottom = std::min(ytop + std::max(1, size_y / 2), half_h_ - 1);
                    if (xright <= xleft || ybottom <= ytop) {
                        x += step_x;
                        continue;
                    }
                    int xmidleft = xleft + (xright - xleft) / 4;
                    int ymidtop = ytop + (ybottom - ytop) / 4;
                    int xmidright = xleft + 3 * (xright - xleft) / 4;
                    int ymidbottom = ytop + 3 * (ybottom - ytop) / 4;
                    int rough_area = std::max(1, static_cast<int>(9.0 * cell_width * cell_height * scale * scale_for_area / 4.0 / 16.0));

                    MotionWindowData wnd = {};
                    wnd.x = xleft * 2;
                    wnd.y = ytop * 2;
                    wnd.width = (xright - xleft) * 2;
                    wnd.height = (ybottom - ytop) * 2;
                    wnd.rough_area = rough_area;
                    wnd.xleft = xleft;
                    wnd.ytop = ytop;
                    wnd.xright = xright;
                    wnd.ybottom = ybottom;
                    wnd.xmidleft = xmidleft;
                    wnd.ymidtop = ymidtop;
                    wnd.xmidright = xmidright;
                    wnd.ymidbottom = ymidbottom;
                    wnd.result_immediate_allowed = true;
                    windows_.push_back(wnd);

                    x += step_x;
                }
                y += step_y;
            }
            size_x = static_cast<int>(size_x * 1.5 + 0.5);
            size_y = static_cast<int>(size_y * 1.5 + 0.5);
            scale *= 1.5;
            scale_for_area *= 1.28;
        }
    }

    inline int64_t sat_at(int sy, int sx) const {
        return sat_[static_cast<size_t>(sy) * (half_w_ + 1) + sx];
    }

    int64_t rect_sum(int x1, int y1, int x2, int y2) const {
        x1 = std::max(0, std::min(x1, half_w_ - 1));
        x2 = std::max(0, std::min(x2, half_w_ - 1));
        y1 = std::max(0, std::min(y1, half_h_ - 1));
        y2 = std::max(0, std::min(y2, half_h_ - 1));
        return sat_at(y2 + 1, x2 + 1) - sat_at(y1, x2 + 1) - sat_at(y2 + 1, x1) + sat_at(y1, x1);
    }

    void build_diff_and_sat(const uint8_t* prev, const uint8_t* curr,
                            int full_w, int full_h, int prev_ch, int curr_ch,
                            size_t prev_stride, size_t curr_stride) {
        // Subsample to half-res and compute diff
        for (int hy = 0; hy < half_h_; ++hy) {
            int fy = hy * 2;
            if (fy >= full_h) fy = full_h - 1;
            for (int hx = 0; hx < half_w_; ++hx) {
                int fx = hx * 2;
                if (fx >= full_w) fx = full_w - 1;
                int pv, cv;
                if (prev_ch == 1) {
                    pv = prev[fy * static_cast<int>(prev_stride) + fx];
                } else {
                    const uint8_t* pp = prev + fy * prev_stride + fx * prev_ch;
                    pv = (29 * pp[0] + 150 * pp[1] + 77 * pp[2]) >> 8;  // BGR to gray
                }
                if (curr_ch == 1) {
                    cv = curr[fy * static_cast<int>(curr_stride) + fx];
                } else {
                    const uint8_t* cp = curr + fy * curr_stride + fx * curr_ch;
                    cv = (29 * cp[0] + 150 * cp[1] + 77 * cp[2]) >> 8;
                }
                int d = pv - cv;
                diff_[static_cast<size_t>(hy) * half_w_ + hx] = static_cast<uint8_t>(d < 0 ? -d : d);
            }
        }
        // 3x3 box blur into temp then copy back
        std::vector<uint8_t> blur(static_cast<size_t>(half_w_) * half_h_);
        for (int hy = 0; hy < half_h_; ++hy) {
            int hy_prev = std::max(0, hy - 1);
            int hy_next = std::min(half_h_ - 1, hy + 1);
            for (int hx = 0; hx < half_w_; ++hx) {
                int hx_prev = std::max(0, hx - 1);
                int hx_next = std::min(half_w_ - 1, hx + 1);
                int sum = 0;
                for (int dy = hy_prev; dy <= hy_next; ++dy)
                    for (int dx = hx_prev; dx <= hx_next; ++dx)
                        sum += diff_[static_cast<size_t>(dy) * half_w_ + dx];
                blur[static_cast<size_t>(hy) * half_w_ + hx] = static_cast<uint8_t>(sum / 9);
            }
        }
        diff_.swap(blur);

        // Integral image: sat[i][j] = sum diff[0..i-1][0..j-1]
        std::memset(sat_.data(), 0, sat_.size() * sizeof(int64_t));
        for (int i = 1; i <= half_h_; ++i) {
            for (int j = 1; j <= half_w_; ++j) {
                int64_t v = diff_[static_cast<size_t>(i - 1) * half_w_ + (j - 1)];
                sat_[static_cast<size_t>(i) * (half_w_ + 1) + j] = v
                    + sat_at(i - 1, j) + sat_at(i, j - 1) - sat_at(i - 1, j - 1);
            }
        }
    }

    bool check_immediate(int x1, int y1, int x2, int y2, int area) const {
        int64_t diffcnt = rect_sum(x1, y1, x2, y2);
        int64_t ratio = diffcnt * 1000;
        return ratio > static_cast<int64_t>(MIN_FILL_RATIO_IMMEDIATE) * area;
    }

    void check_quadrant(int x1, int y1, int x2, int y2, int area, int q,
                       MotionWindowData& wnd, bool& result, bool& result_fast) const {
        int64_t diffcnt = rect_sum(x1, y1, x2, y2);
        int64_t ratio = diffcnt * 1000;
        int& cnt = wnd.motion_cnt[q];
        cnt -= cnt / MOTION_CNT_DROP;
        if (ratio > static_cast<int64_t>(MIN_FILL_RATIO) * area)
            cnt += MOTION_CNT_ADD;
        cnt = std::max(0, std::min(32767, cnt));
        if (cnt > MOTION_CNT_LIMIT) result = true;
        if (ratio > static_cast<int64_t>(MIN_FILL_RATIO_FAST) * area) result_fast = true;
    }

    int update_windows(int frame_time_ms, bool can_process, bool do_reset) {
        int active_count = 0;
        for (auto& wnd : windows_) {
            bool result = false;
            bool result_fast_actual = false;
            bool result_immediate = false;
            int area = wnd.rough_area;

            if (can_process && !do_reset) {
                result_immediate = check_immediate(wnd.xleft, wnd.ytop, wnd.xmidright, wnd.ymidbottom, area)
                    || check_immediate(wnd.xmidleft, wnd.ytop, wnd.xright, wnd.ymidbottom, area)
                    || check_immediate(wnd.xleft, wnd.ymidtop, wnd.xmidright, wnd.ybottom, area)
                    || check_immediate(wnd.xmidleft, wnd.ymidtop, wnd.xright, wnd.ybottom, area);
                check_quadrant(wnd.xleft, wnd.ytop, wnd.xmidright, wnd.ymidbottom, area, 0, wnd, result, result_fast_actual);
                check_quadrant(wnd.xmidleft, wnd.ytop, wnd.xright, wnd.ymidbottom, area, 1, wnd, result, result_fast_actual);
                check_quadrant(wnd.xleft, wnd.ymidtop, wnd.xmidright, wnd.ybottom, area, 2, wnd, result, result_fast_actual);
                check_quadrant(wnd.xmidleft, wnd.ymidtop, wnd.xright, wnd.ybottom, area, 3, wnd, result, result_fast_actual);
            }

            if (result) {
                wnd.frames_from_motion = std::min(wnd.frames_from_motion + FRAMES_FROM_MOTION_ADD, FRAMES_FROM_MOTION_LIMIT);
            }
            if (result_fast_actual) {
                wnd.frames_from_motion_fast = std::min(wnd.frames_from_motion_fast + FRAMES_FROM_MOTION_ADD_FAST, FRAMES_FROM_MOTION_LIMIT_FAST);
            }

            if (wnd.frames_from_motion > 0) {
                if (wnd.motion_state == 0) {
                    wnd.frames_by_immediate_tests = 0;
                    wnd.frames_by_immediate = 0;
                    wnd.motion_state = 1;
                    wnd.result_immediate_allowed = true;
                }
                if (wnd.motion_state == 1) {
                    if (result_immediate) {
                        if (frame_time_ms > MOTION_VALIDATION_FPS_MS_LIMIT)
                            wnd.result_immediate_allowed = true;
                        if (wnd.result_immediate_allowed)
                            wnd.frames_by_immediate += 1;
                        wnd.result_immediate_allowed = !wnd.result_immediate_allowed;
                    } else {
                        wnd.result_immediate_allowed = true;
                    }
                    if (wnd.frames_by_immediate >= FRAME_CNT_MOTION_VALIDATED) {
                        wnd.motion_state = 2;
                        wnd.frames_by_immediate_tests = 0;
                    } else {
                        wnd.frames_by_immediate_tests += 1;
                    }
                }
                if (wnd.frames_by_immediate_tests >= FRAME_CNT_MOTION_NOT_VALIDATED) {
                    wnd.frames_from_motion = std::max(0, wnd.frames_from_motion - FRAMES_FROM_MOTION_ADD / 2);
                } else {
                    wnd.frames_from_motion = std::max(0, wnd.frames_from_motion - 1);
                }
                if (wnd.frames_from_motion <= 0)
                    wnd.motion_state = 0;
            }

            if (wnd.frames_from_motion_fast > 0) {
                wnd.frames_from_motion_fast = std::max(0, wnd.frames_from_motion_fast - 4);
            }

            wnd.motion_super_fast = result_fast_actual ? 1 : 0;
            wnd.activate_off_by_motion = result_fast_actual || wnd.frames_from_motion_fast > 0 || wnd.frames_from_motion > 0;
            if (wnd.activate_off_by_motion)
                active_count += 1;
        }
        return active_count;
    }
};

void bind_motion(py::module_& m) {
    py::class_<MotionEngine>(m, "MotionEngine")
        .def(py::init<int, int, int, int, int>(),
             py::arg("frame_width"), py::arg("frame_height"),
             py::arg("cell_width") = 25, py::arg("cell_height") = 25, py::arg("noof_zooms") = 5)
        .def("get_window_count", &MotionEngine::get_window_count)
        .def("process_motion", &MotionEngine::process_motion,
             py::arg("prev_frame"), py::arg("curr_frame"),
             py::arg("frame_time_ms"), py::arg("can_process") = true, py::arg("do_reset") = false)
        .def("check_triggered", &MotionEngine::check_triggered,
             py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"));
}
