#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>

#include "motion_cpp.h"

namespace py = pybind11;

// Helper to create meshgrid-like grid values
static void make_grid(int nx, int ny, int na, std::vector<float>& grid_x, std::vector<float>& grid_y) {
    grid_x.resize(na * ny * nx);
    grid_y.resize(na * ny * nx);
    for (int a = 0; a < na; ++a) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int idx = a * (ny * nx) + y * nx + x;
                grid_x[idx] = (float)x;
                grid_y[idx] = (float)y;
            }
        }
    }
}

py::array_t<float> detect_forward(py::list xs, py::array_t<float> anchors_arr, int nc, std::vector<int> strides) {
    // anchors_arr: (nl, na, 2)
    auto anchors_buf = anchors_arr.unchecked<3>();
    int nl = anchors_buf.shape(0);
    int na = anchors_buf.shape(1);
    int no = nc + 5;

    // Prepare per-layer outputs
    std::vector< std::vector<float> > layer_z;
    std::vector<int> layer_counts;
    int bs = -1;

    for (int i = 0; i < nl; ++i) {
        py::array_t<float> xi = xs[i].cast<py::array_t<float>>();
        auto buf = xi.unchecked<4>(); // (bs, ch, ny, nx)
        int bsi = buf.shape(0);
        int ch = buf.shape(1);
        int ny = buf.shape(2);
        int nx = buf.shape(3);

        if (bs == -1) bs = bsi;

        // Expected ch == na * no
        // Precompute grid and anchor_grid
        std::vector<float> grid_x, grid_y;
        make_grid(nx, ny, na, grid_x, grid_y);

        float stride = (i < (int)strides.size()) ? (float)strides[i] : 32.0f;

        // anchor grid values per anchor (a, 2)
        std::vector<float> anchor_xy(na * 2);
        for (int a = 0; a < na; ++a) {
            anchor_xy[a*2 + 0] = anchors_buf(i, a, 0) * stride;
            anchor_xy[a*2 + 1] = anchors_buf(i, a, 1) * stride;
        }

        int layer_preds = na * ny * nx;
        std::vector<float> z_layer(bs * layer_preds * no);

        // Iterate and fill z_layer
        for (int b = 0; b < bs; ++b) {
            for (int a = 0; a < na; ++a) {
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        int pred_index = ((a * ny + y) * nx + x);
                        int out_base = (b * layer_preds + pred_index) * no;
                        for (int k = 0; k < no; ++k) {
                            int c = a * no + k; // channel index
                            float v = buf(b, c, y, x);
                            float outv = v;
                            if (k == 0) {
                                // x center
                                outv = (v + 0.5f + grid_x[a*ny*nx + y*nx + x]) * stride;
                            } else if (k == 1) {
                                // y center
                                outv = (v + 0.5f + grid_y[a*ny*nx + y*nx + x]) * stride;
                            } else if (k == 2) {
                                // width
                                float tmp = v + 1.0f;
                                outv = tmp * tmp * anchor_xy[a*2 + 0];
                            } else if (k == 3) {
                                // height
                                float tmp = v + 1.0f;
                                outv = tmp * tmp * anchor_xy[a*2 + 1];
                            } else {
                                // confidence and classes
                                outv = v * 0.5f + 0.5f;
                            }
                            z_layer[out_base + k] = outv;
                        }
                    }
                }
            }
        }

        layer_counts.push_back(layer_preds);
        layer_z.push_back(std::move(z_layer));
    }

    // Concatenate layers along the second dimension per batch
    int total_preds = 0;
    for (int c : layer_counts) total_preds += c;

    std::vector<float> out(bs * total_preds * no);
    for (int b = 0; b < bs; ++b) {
        int pred_offset = 0;
        for (size_t li = 0; li < layer_z.size(); ++li) {
            int layer_preds = layer_counts[li];
            const std::vector<float>& z = layer_z[li];
            const float* src = z.data() + b * layer_preds * no;
            float* dst = out.data() + (b * total_preds + pred_offset) * no;
            size_t bytes = (size_t)layer_preds * (size_t)no;
            for (size_t t = 0; t < bytes; ++t) dst[t] = src[t];
            pred_offset += layer_preds;
        }
    }

    // Return as numpy array with shape (bs, total_preds, no)
    std::vector<ssize_t> shape = { bs, total_preds, no };
    return py::array_t<float>(shape, out.data());
}

PYBIND11_MODULE(cpp_detect, m) {
    m.doc() = "Optimized Detect forward";
    m.def("detect_forward", &detect_forward, "Detect forward", py::arg("xs"), py::arg("anchors"), py::arg("nc"), py::arg("strides"));
    m.def("nms", [](py::array_t<float> boxes_arr, py::array_t<float> scores_arr, float iou_thres) {
        auto boxes = boxes_arr.unchecked<2>(); // (N,4)
        auto scores = scores_arr.unchecked<1>(); // (N,)
        int N = boxes.shape(0);

        std::vector<float> x1(N), y1(N), x2(N), y2(N), areas(N);
        for (int i = 0; i < N; ++i) {
            x1[i] = boxes(i, 0);
            y1[i] = boxes(i, 1);
            x2[i] = boxes(i, 2);
            y2[i] = boxes(i, 3);
            areas[i] = (x2[i] - x1[i]) * (y2[i] - y1[i]);
        }

        // indices sorted by score desc
        std::vector<int> idxs(N);
        for (int i = 0; i < N; ++i) idxs[i] = i;
        std::sort(idxs.begin(), idxs.end(), [&](int a, int b){ return scores(a) > scores(b); });

        std::vector<int> keep;
        while (!idxs.empty()) {
            int i = idxs[0];
            keep.push_back(i);
            std::vector<int> rem;
            for (size_t k = 1; k < idxs.size(); ++k) {
                int j = idxs[k];
                float xx1 = std::max(x1[i], x1[j]);
                float yy1 = std::max(y1[i], y1[j]);
                float xx2 = std::min(x2[i], x2[j]);
                float yy2 = std::min(y2[i], y2[j]);
                float w = std::max(0.0f, xx2 - xx1);
                float h = std::max(0.0f, yy2 - yy1);
                float inter = w * h;
                float ovr = inter / (areas[i] + areas[j] - inter);
                if (ovr <= iou_thres) rem.push_back(j);
            }
            idxs = std::move(rem);
        }

        // return as numpy array of int64
        py::array_t<long long> out(keep.size());
        auto buf = out.mutable_unchecked<1>();
        for (size_t i = 0; i < keep.size(); ++i) buf(i) = (long long)keep[i];
        return out;
    }, "Greedy NMS (boxes: Nx4 xyxy, scores: N)", py::arg("boxes"), py::arg("scores"), py::arg("iou_thres"));

    // HPD2 adaptive thresholding: vectorized per-window thresholding
    m.def("apply_hpd2_threshold", [](py::array_t<int> widths_arr,
                                       py::array_t<int> heights_arr,
                                       py::array_t<int> y_frombottom_arr,
                                       py::array_t<int> values_arr,
                                       int image_height,
                                       int int_threshold) {
        auto widths = widths_arr.unchecked<1>();
        auto heights = heights_arr.unchecked<1>();
        auto y_frombottom = y_frombottom_arr.unchecked<1>();
        auto values = values_arr.mutable_unchecked<1>();

        ssize_t n = widths.shape(0);
        std::vector<int> selected_indices;
        selected_indices.reserve(n);

        const int SHIFT = 10; // 1<<10

        for (ssize_t i = 0; i < n; ++i) {
            int w = widths(i);
            int h = heights(i);
            int yfb = y_frombottom(i);
            int val = values(i);

            int relative_size = (((w + h) / 2) * (1 << SHIFT)) / image_height;
            int relative_pos = ((yfb + h / 2) * (1 << SHIFT)) / image_height;

            int thresh_ratio1 = std::min(relative_size * 2 + 700, 1024);

            // avoid divide by zero
            int denom = relative_pos - 500 + 800;
            if (denom == 0) denom = 1;
            int thresh_ratio2 = std::max(((1100) << SHIFT) / denom, 1024);

            int thresh_ratio = std::min((thresh_ratio1 * thresh_ratio2) >> SHIFT, 1024);
            int adapted_threshold = (int_threshold * thresh_ratio) >> SHIFT;

            bool pass = (val >= adapted_threshold);
            if (pass) {
                // update low confidence to int_threshold
                if (val >= adapted_threshold && val < int_threshold) {
                    values(i) = int_threshold;
                }
                selected_indices.push_back((int)i);
            }
        }

        // return selected indices as numpy array
        py::array_t<int> out(selected_indices.size());
        auto out_un = out.mutable_unchecked<1>();
        for (size_t i = 0; i < selected_indices.size(); ++i) out_un(i) = selected_indices[i];
        return out;
    }, "Apply HPD2 adaptive thresholding\n(widths, heights, y_frombottom, values, image_height, int_threshold) -> indices");

    // Custom suppress algorithm selection: accepts boxes (Nx4 xyxy) and values (N)
    m.def("suppress_custom_selection", [](py::array_t<int> boxes_arr, py::array_t<long long> values_arr, int max_windows) {
        auto boxes = boxes_arr.unchecked<2>(); // (N,4)
        auto values = values_arr.unchecked<1>();
        int N = boxes.shape(0);
        if (N == 0) return py::array_t<int>(0);

        struct Win { int idx; int x1,y1,x2,y2; int area; long long val; };
        std::vector<Win> wins; wins.reserve(N);
        for (int i = 0; i < N; ++i) {
            Win w; w.idx = i; w.x1 = boxes(i,0); w.y1 = boxes(i,1); w.x2 = boxes(i,2); w.y2 = boxes(i,3);
            w.area = (w.x2 - w.x1) * (w.y2 - w.y1);
            w.val = values(i);
            wins.push_back(w);
        }

        // sort by val desc
        std::sort(wins.begin(), wins.end(), [](const Win &a, const Win &b){ return a.val > b.val; });

        if ((int)wins.size() > max_windows) wins.resize(max_windows);
        int M = wins.size();

        // Build overlap matrix W (M x M) as uint8
        std::vector<uint8_t> W(M * M, 0);

        for (int i = 0; i < M; ++i) {
            for (int j = i+1; j < M; ++j) {
                int xx1 = std::max(wins[i].x1, wins[j].x1);
                int yy1 = std::max(wins[i].y1, wins[j].y1);
                int xx2 = std::min(wins[i].x2, wins[j].x2);
                int yy2 = std::min(wins[i].y2, wins[j].y2);
                int w = std::max(0, xx2 - xx1);
                int h = std::max(0, yy2 - yy1);
                int inter = w * h;
                int min_area = std::min(wins[i].area, wins[j].area);
                bool overlap = inter > (min_area * 22 / 64);
                if (overlap) {
                    W[i*M + j] = 1;
                    W[j*M + i] = 1;
                }
            }
        }

        const long long WALLM_DENSITY_WEIGHT = 1000000LL;

        std::vector<uint8_t> x_active(M, 0);
        // xt_m is int64 matrix MxM initialized from W
        std::vector<long long> xt_m(M * M, 0);
        for (int i = 0; i < M; ++i) for (int j = 0; j < M; ++j) xt_m[i*M + j] = W[i*M + j];

        std::vector<long long> win_value(M);
        for (int i = 0; i < M; ++i) win_value[i] = wins[i].val;

        for (int iter = 0; iter < 1000; ++iter) {
            int x_idx = -1;
            long long cost = LLONG_MAX;

            for (int i = 0; i < M; ++i) {
                if (x_active[i]) continue;
                x_active[i] = 1;

                // compute x0 = -dot(win_value, x_active)
                long long dot = 0;
                for (int k = 0; k < M; ++k) if (x_active[k]) dot += win_value[k];
                long long x0 = -dot;

                // x1 = dot(xt_m[i], x_active) * WALLM_DENSITY_WEIGHT
                long long dot2 = 0;
                for (int k = 0; k < M; ++k) if (x_active[k]) dot2 += xt_m[i*M + k];
                long long x1 = dot2 * WALLM_DENSITY_WEIGHT;

                x_active[i] = 0;

                if (x0 + x1 < cost) {
                    cost = x0 + x1;
                    x_idx = i;
                }
            }

            if (x_idx >= 0) {
                x_active[x_idx] = 1;
                // xt_m += W[x_idx][:, np.newaxis] => add column to each row
                for (int r = 0; r < M; ++r) {
                    xt_m[r*M + x_idx] += W[r*M + x_idx];
                }
            } else break;
        }

        // Collect active original indices
        std::vector<int> out_idx;
        for (int i = 0; i < M; ++i) if (x_active[i]) out_idx.push_back(wins[i].idx);

        py::array_t<int> out(out_idx.size());
        auto out_un = out.mutable_unchecked<1>();
        for (size_t i = 0; i < out_idx.size(); ++i) out_un(i) = out_idx[i];
        return out;
    }, "Custom suppress/selection (boxes Nx4 int, values N, max_windows) -> selected original indices");

    // Convert detections -> arrays (x,y,w,h,value)
    m.def("detections_to_arrays", [](py::array_t<float> detections_arr, int image_height) {
        auto det = detections_arr.unchecked<2>(); // (M, >=6)
        int M = det.shape(0);
        if (M == 0) return py::make_tuple(py::array_t<int>(0), py::array_t<int>(0), py::array_t<int>(0), py::array_t<int>(0), py::array_t<int>(0));

        py::array_t<int> xs(M), ys(M), ws(M), hs(M), vals(M);
        auto xs_un = xs.mutable_unchecked<1>();
        auto ys_un = ys.mutable_unchecked<1>();
        auto ws_un = ws.mutable_unchecked<1>();
        auto hs_un = hs.mutable_unchecked<1>();
        auto vals_un = vals.mutable_unchecked<1>();

        for (int i = 0; i < M; ++i) {
            float x1 = det(i,0);
            float y1 = det(i,1);
            float x2 = det(i,2);
            float y2 = det(i,3);
            float conf = det(i,4);

            int xi = (int)std::round(x1);
            int yi = (int)std::round(y1);
            int wi = (int)std::round(x2 - x1);
            int hi = (int)std::round(y2 - y1);
            int value = (int)std::round(conf * 5000.0f * 100.0f);

            xs_un(i) = xi;
            ys_un(i) = yi;
            ws_un(i) = wi;
            hs_un(i) = hi;
            vals_un(i) = value;
        }

        return py::make_tuple(xs, ys, ws, hs, vals);
    }, "Convert detections (M x >=6) to arrays x,y,w,h,value");

    bind_motion(m);
}
