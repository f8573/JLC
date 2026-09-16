#include "jlc_native.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace {

constexpr long double JACOBI_TOL =
    64.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());
constexpr double QR_TOL = 8.0 * std::numeric_limits<double>::epsilon();

struct ColumnPairStats {
    long double pp;
    long double qq;
    long double pq;
};

ColumnPairStats column_pair_stats(const std::vector<double>& work,
                                  int rows, int cols, int p, int q) {
    long double pp = 0.0L;
    long double qq = 0.0L;
    long double pq = 0.0L;
    for (int row = 0; row < rows; ++row) {
        const double x = work[static_cast<std::size_t>(row) * cols + p];
        const double y = work[static_cast<std::size_t>(row) * cols + q];
        pp += static_cast<long double>(x) * static_cast<long double>(x);
        qq += static_cast<long double>(y) * static_cast<long double>(y);
        pq += static_cast<long double>(x) * static_cast<long double>(y);
    }
    return {pp, qq, pq};
}

double column_norm(const std::vector<double>& values, int rows, int cols, int col) {
    double scale = 0.0;
    long double sum = 1.0L;
    for (int row = 0; row < rows; ++row) {
        const double value = std::abs(values[static_cast<std::size_t>(row) * cols + col]);
        if (value == 0.0) {
            continue;
        }
        if (scale < value) {
            const long double ratio = scale == 0.0
                ? 0.0L
                : static_cast<long double>(scale) / static_cast<long double>(value);
            sum = 1.0L + sum * ratio * ratio;
            scale = value;
        } else {
            const long double ratio = static_cast<long double>(value)
                / static_cast<long double>(scale);
            sum += ratio * ratio;
        }
    }
    if (scale == 0.0) {
        return 0.0;
    }
    const long double result = static_cast<long double>(scale) * std::sqrt(sum);
    return static_cast<double>(result);
}

void set_identity(double* out, int n) {
    std::fill(out, out + static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i) * n + i] = 1.0;
    }
}

void rotate_columns_jacobi(std::vector<double>& work, int rows, int cols,
                           int p, int q, double c, double s) {
    for (int row = 0; row < rows; ++row) {
        const std::size_t p_index = static_cast<std::size_t>(row) * cols + p;
        const std::size_t q_index = static_cast<std::size_t>(row) * cols + q;
        const double x = work[p_index];
        const double y = work[q_index];
        work[p_index] = c * x - s * y;
        work[q_index] = s * x + c * y;
    }
}

bool complete_left_vectors(const std::vector<double>& work,
                           int rows, int cols,
                           const std::vector<int>& order,
                           const std::vector<double>& norms,
                           double* u) {
    std::fill(u, u + static_cast<std::size_t>(rows) * static_cast<std::size_t>(rows), 0.0);
    const int rank_slots = std::min(rows, cols);
    const double completion_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() * std::max(1, rows);
    std::vector<double> candidate(static_cast<std::size_t>(rows), 0.0);
    int next_basis = 0;

    for (int col = 0; col < rows; ++col) {
        const int source = col < rank_slots ? order[static_cast<std::size_t>(col)] : -1;
        const double sigma = source >= 0 ? norms[static_cast<std::size_t>(source)] : 0.0;
        if (source >= 0 && sigma > 0.0 && std::isfinite(sigma)) {
            for (int row = 0; row < rows; ++row) {
                candidate[static_cast<std::size_t>(row)] =
                    work[static_cast<std::size_t>(row) * cols + source] / sigma;
            }
            const double candidate_norm = column_norm(candidate, rows, 1, 0);
            if (!(candidate_norm > 0.0) || !std::isfinite(candidate_norm)) {
                return false;
            }
            for (int row = 0; row < rows; ++row) {
                candidate[static_cast<std::size_t>(row)] /= candidate_norm;
                u[static_cast<std::size_t>(row) * rows + col] =
                    candidate[static_cast<std::size_t>(row)];
            }
            continue;
        }

        bool accepted = false;
        for (int basis = next_basis; basis < rows && !accepted; ++basis) {
            std::fill(candidate.begin(), candidate.end(), 0.0);
            candidate[static_cast<std::size_t>(basis)] = 1.0;

            for (int pass = 0; pass < 2; ++pass) {
                for (int previous = 0; previous < col; ++previous) {
                    long double dot = 0.0L;
                    for (int row = 0; row < rows; ++row) {
                        dot += static_cast<long double>(candidate[static_cast<std::size_t>(row)])
                            * static_cast<long double>(
                                u[static_cast<std::size_t>(row) * rows + previous]);
                    }
                    for (int row = 0; row < rows; ++row) {
                        candidate[static_cast<std::size_t>(row)] -=
                            static_cast<double>(dot)
                            * u[static_cast<std::size_t>(row) * rows + previous];
                    }
                }
            }

            const double candidate_norm = column_norm(candidate, rows, 1, 0);
            if (candidate_norm > completion_tolerance && std::isfinite(candidate_norm)) {
                for (int row = 0; row < rows; ++row) {
                    candidate[static_cast<std::size_t>(row)] /= candidate_norm;
                    u[static_cast<std::size_t>(row) * rows + col] =
                        candidate[static_cast<std::size_t>(row)];
                }
                next_basis = basis + 1;
                accepted = true;
            }
        }
        if (!accepted) {
            return false;
        }
    }
    return true;
}

/* The one-sided Jacobi core is retained as the small-matrix native algorithm. */
jlc_status svd_jacobi_tall(const double* a, int m, int n,
                           double* u, double* singular_values, int singular_count,
                           double* v) {
    if (a == nullptr || u == nullptr || singular_values == nullptr || v == nullptr
        || m < 0 || n < 0 || singular_count != std::min(m, n)) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0 || n == 0) {
        set_identity(u, m);
        set_identity(v, n);
        return JLC_STATUS_SUCCESS;
    }

    try {
        const std::size_t work_size = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        const std::size_t right_size = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
        std::vector<double> work(a, a + work_size);
        std::vector<double> right(right_size, 0.0);
        for (int i = 0; i < n; ++i) {
            right[static_cast<std::size_t>(i) * n + i] = 1.0;
        }

        for (double value : work) {
            if (!std::isfinite(value)) {
                return JLC_STATUS_INVALID_ARGUMENT;
            }
        }

        const int max_dim = std::max(m, n);
        const long long proposed_sweeps = 8LL * static_cast<long long>(max_dim);
        const int max_sweeps = static_cast<int>(std::min<long long>(
            2048LL, std::max<long long>(20LL, proposed_sweeps)));
        bool converged = false;

        // One-sided Jacobi diagonalizes the column Gram matrix implicitly.
        // The same right rotations are accumulated in V.
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            bool rotated = false;
            for (int p = 0; p < n - 1; ++p) {
                for (int q = p + 1; q < n; ++q) {
                    const ColumnPairStats stats = column_pair_stats(work, m, n, p, q);
                    if (!(stats.pp > 0.0L) || !(stats.qq > 0.0L)
                        || !std::isfinite(stats.pp) || !std::isfinite(stats.qq)
                        || !std::isfinite(stats.pq)) {
                        continue;
                    }
                    const long double scale = std::sqrt(stats.pp) * std::sqrt(stats.qq);
                    if (!(scale > 0.0L) || std::abs(stats.pq) <= JACOBI_TOL * scale) {
                        continue;
                    }

                    const long double tau = (stats.qq - stats.pp) / (2.0L * stats.pq);
                    const long double t = (tau >= 0.0L ? 1.0L : -1.0L)
                        / (std::abs(tau) + std::hypot(1.0L, tau));
                    const long double c_long = 1.0L / std::sqrt(1.0L + t * t);
                    const double c = static_cast<double>(c_long);
                    const double s = static_cast<double>(t * c_long);
                    if (!std::isfinite(c) || !std::isfinite(s)) {
                        return JLC_STATUS_INTERNAL_ERROR;
                    }

                    rotate_columns_jacobi(work, m, n, p, q, c, s);
                    rotate_columns_jacobi(right, n, n, p, q, c, s);
                    rotated = true;
                }
            }
            if (!rotated) {
                converged = true;
                break;
            }
        }

        if (!converged) {
            return JLC_STATUS_CONVERGENCE_FAILURE;
        }

        std::vector<double> norms(static_cast<std::size_t>(n), 0.0);
        for (int col = 0; col < n; ++col) {
            norms[static_cast<std::size_t>(col)] = column_norm(work, m, n, col);
            if (!std::isfinite(norms[static_cast<std::size_t>(col)])) {
                return JLC_STATUS_INTERNAL_ERROR;
            }
        }

        std::vector<int> order(static_cast<std::size_t>(n), 0);
        for (int col = 0; col < n; ++col) {
            order[static_cast<std::size_t>(col)] = col;
        }
        std::stable_sort(order.begin(), order.end(), [&norms](int left, int right_index) {
            return norms[static_cast<std::size_t>(left)]
                > norms[static_cast<std::size_t>(right_index)];
        });

        const int rank_slots = std::min(m, n);
        for (int i = 0; i < rank_slots; ++i) {
            singular_values[i] = norms[static_cast<std::size_t>(
                order[static_cast<std::size_t>(i)])];
        }

        std::fill(v, v + right_size, 0.0);
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                v[static_cast<std::size_t>(row) * n + col] =
                    right[static_cast<std::size_t>(row) * n
                          + order[static_cast<std::size_t>(col)]];
            }
        }

        if (!complete_left_vectors(work, m, n, order, norms, u)) {
            return JLC_STATUS_INTERNAL_ERROR;
        }
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
}

jlc_status svd_jacobi_unscaled(const double* a, int m, int n,
                               double* u, double* singular_values, int singular_count,
                               double* v) {
    if (m >= n) {
        return svd_jacobi_tall(a, m, n, u, singular_values, singular_count, v);
    }

    try {
        const std::size_t input_size = static_cast<std::size_t>(m)
            * static_cast<std::size_t>(n);
        const std::size_t left_size = static_cast<std::size_t>(n)
            * static_cast<std::size_t>(n);
        const std::size_t right_size = static_cast<std::size_t>(m)
            * static_cast<std::size_t>(m);
        std::vector<double> transposed(input_size, 0.0);
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                transposed[static_cast<std::size_t>(col) * m + row] =
                    a[static_cast<std::size_t>(row) * n + col];
            }
        }
        std::vector<double> u_t(left_size, 0.0);
        std::vector<double> v_t(right_size, 0.0);
        const jlc_status status = svd_jacobi_tall(
            transposed.data(), n, m, u_t.data(), singular_values, singular_count, v_t.data());
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
        std::copy(v_t.begin(), v_t.end(), u);
        std::copy(u_t.begin(), u_t.end(), v);
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
}

bool normalize_input(const double* input, int rows, int cols,
                     std::vector<double>& normalized, double& scale) {
    const std::size_t size = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(cols);
    normalized.resize(size);
    scale = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        const double value = input[i];
        if (!std::isfinite(value)) {
            return false;
        }
        scale = std::max(scale, std::abs(value));
    }
    if (scale == 0.0) {
        std::fill(normalized.begin(), normalized.end(), 0.0);
        return true;
    }
    for (std::size_t i = 0; i < size; ++i) {
        normalized[i] = input[i] / scale;
    }
    return true;
}

bool rescale_singular_values(double* singular_values, int count, double scale) {
    for (int i = 0; i < count; ++i) {
        const double value = singular_values[i] * scale;
        if (!std::isfinite(value) || value < 0.0) {
            return false;
        }
        singular_values[i] = value;
    }
    return true;
}

jlc_status svd_jacobi_scaled(const double* a, int m, int n,
                             double* u, double* singular_values, int singular_count,
                             double* v) {
    try {
        std::vector<double> normalized;
        double scale = 0.0;
        if (!normalize_input(a, m, n, normalized, scale)) {
            return JLC_STATUS_INVALID_ARGUMENT;
        }
        if (scale == 0.0) {
            set_identity(u, m);
            set_identity(v, n);
            std::fill(singular_values, singular_values + singular_count, 0.0);
            return JLC_STATUS_SUCCESS;
        }
        const jlc_status status = svd_jacobi_unscaled(
            normalized.data(), m, n, u, singular_values, singular_count, v);
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
        return rescale_singular_values(singular_values, singular_count, scale)
            ? JLC_STATUS_SUCCESS : JLC_STATUS_INTERNAL_ERROR;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
}

struct GivensRotation {
    double c;
    double s;
    double r;
};

GivensRotation make_givens(double f, double g) {
    const double r = std::hypot(f, g);
    if (r == 0.0 || !std::isfinite(r)) {
        return {1.0, 0.0, r};
    }
    return {f / r, g / r, r};
}

void apply_givens_to_columns(double* matrix, int rows, int stride,
                             int first, int second, double c, double s) {
    for (int row = 0; row < rows; ++row) {
        const std::size_t first_index = static_cast<std::size_t>(row) * stride + first;
        const std::size_t second_index = static_cast<std::size_t>(row) * stride + second;
        const double x = matrix[first_index];
        const double y = matrix[second_index];
        matrix[first_index] = c * x + s * y;
        matrix[second_index] = -s * x + c * y;
    }
}

void negate_column(double* matrix, int rows, int stride, int column) {
    for (int row = 0; row < rows; ++row) {
        matrix[static_cast<std::size_t>(row) * stride + column] =
            -matrix[static_cast<std::size_t>(row) * stride + column];
    }
}

void swap_columns(double* matrix, int rows, int stride, int first, int second) {
    if (first == second) {
        return;
    }
    for (int row = 0; row < rows; ++row) {
        const std::size_t first_index = static_cast<std::size_t>(row) * stride + first;
        const std::size_t second_index = static_cast<std::size_t>(row) * stride + second;
        std::swap(matrix[first_index], matrix[second_index]);
    }
}

double max_abs4(double a, double b, double c, double d) {
    return std::max(std::max(std::abs(a), std::abs(b)),
                    std::max(std::abs(c), std::abs(d)));
}

double qr_deflation_threshold(double a, double b) {
    return QR_TOL * (std::abs(a) + std::abs(b))
        + std::numeric_limits<double>::denorm_min();
}

double qr_zero_diagonal_threshold(double left_off, double right_off) {
    return QR_TOL * (std::abs(left_off) + std::abs(right_off))
        + std::numeric_limits<double>::denorm_min();
}

/*
 * Reduce an upper bidiagonal matrix to diagonal form with implicit shifted
 * Golub-Kahan QR steps. The left and right factors are already the accumulated
 * Householder factors from jlc_native_bidiagonal_decompose; rotations are
 * applied directly to their columns, so no second full-factor multiply is
 * needed.
 */
jlc_status bidiagonal_qr(const double* b, int rows, int cols,
                         double* left_factor, double* right_factor,
                         double* singular_values) {
    const int p = std::min(rows, cols);
    if (p == 0) {
        return JLC_STATUS_SUCCESS;
    }

    std::vector<double> s(static_cast<std::size_t>(p), 0.0);
    std::vector<double> e(static_cast<std::size_t>(p), 0.0);
    for (int i = 0; i < p; ++i) {
        s[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(i) * cols + i];
        if (i + 1 < p) {
            e[static_cast<std::size_t>(i)] =
                b[static_cast<std::size_t>(i) * cols + i + 1];
            if (!std::isfinite(e[static_cast<std::size_t>(i)])) {
                return JLC_STATUS_INTERNAL_ERROR;
            }
        }
        if (!std::isfinite(s[static_cast<std::size_t>(i)])) {
            return JLC_STATUS_INTERNAL_ERROR;
        }
    }

    const int max_iterations = std::max(64, 30 * p);
    int iterations = 0;
    int pp = p;
    int exceptional_count = 0;
    int zero_shift_iterations = 0;

    while (pp > 0) {
        if (iterations++ >= max_iterations) {
            return JLC_STATUS_CONVERGENCE_FAILURE;
        }

        int k = pp - 2;
        for (; k >= -1; --k) {
            if (k == -1) {
                break;
            }
            if (std::abs(e[static_cast<std::size_t>(k)])
                <= qr_deflation_threshold(s[static_cast<std::size_t>(k)],
                                           s[static_cast<std::size_t>(k + 1)])) {
                e[static_cast<std::size_t>(k)] = 0.0;
                break;
            }
        }

        int kase = 0;
        if (k == pp - 2) {
            kase = 4;
        } else {
            int ks = pp - 1;
            for (; ks >= k; --ks) {
                if (ks == k) {
                    break;
                }
                const double left_off = ks > 0 ? e[static_cast<std::size_t>(ks - 1)] : 0.0;
                const double right_off = ks < p ? e[static_cast<std::size_t>(ks)] : 0.0;
                if (std::abs(s[static_cast<std::size_t>(ks)])
                    <= qr_zero_diagonal_threshold(left_off, right_off)) {
                    s[static_cast<std::size_t>(ks)] = 0.0;
                    break;
                }
            }
            if (ks == k) {
                kase = 3;
            } else if (ks == pp - 1) {
                kase = 1;
            } else {
                kase = 2;
                k = ks;
            }
        }
        ++k;

        switch (kase) {
            case 1: {
                double f = e[static_cast<std::size_t>(pp - 2)];
                e[static_cast<std::size_t>(pp - 2)] = 0.0;
                for (int j = pp - 2; j >= k; --j) {
                    const GivensRotation rotation = make_givens(
                        s[static_cast<std::size_t>(j)], f);
                    if (!std::isfinite(rotation.r)) {
                        return JLC_STATUS_INTERNAL_ERROR;
                    }
                    s[static_cast<std::size_t>(j)] = rotation.r;
                    if (j != k) {
                        f = -rotation.s * e[static_cast<std::size_t>(j - 1)];
                        e[static_cast<std::size_t>(j - 1)] =
                            rotation.c * e[static_cast<std::size_t>(j - 1)];
                    }
                    apply_givens_to_columns(right_factor, cols, cols, j, pp - 1,
                                            rotation.c, rotation.s);
                }
                exceptional_count = 0;
                break;
            }
            case 2: {
                double f = e[static_cast<std::size_t>(k - 1)];
                e[static_cast<std::size_t>(k - 1)] = 0.0;
                for (int j = k; j < pp; ++j) {
                    const GivensRotation rotation = make_givens(
                        s[static_cast<std::size_t>(j)], f);
                    if (!std::isfinite(rotation.r)) {
                        return JLC_STATUS_INTERNAL_ERROR;
                    }
                    s[static_cast<std::size_t>(j)] = rotation.r;
                    f = -rotation.s * e[static_cast<std::size_t>(j)];
                    e[static_cast<std::size_t>(j)] =
                        rotation.c * e[static_cast<std::size_t>(j)];
                    apply_givens_to_columns(left_factor, rows, rows, j, k - 1,
                                            rotation.c, rotation.s);
                }
                exceptional_count = 0;
                break;
            }
            case 3: {
                ++exceptional_count;
                const double local_scale = max_abs4(
                    s[static_cast<std::size_t>(pp - 1)],
                    s[static_cast<std::size_t>(pp - 2)],
                    e[static_cast<std::size_t>(pp - 2)],
                    s[static_cast<std::size_t>(k)]);
                const double scale = std::max(
                    local_scale, std::abs(e[static_cast<std::size_t>(k)]));
                const double safe_scale = scale == 0.0 ? 1.0 : scale;
                const double sp = s[static_cast<std::size_t>(pp - 1)] / safe_scale;
                const double spm1 = s[static_cast<std::size_t>(pp - 2)] / safe_scale;
                const double epm1 = e[static_cast<std::size_t>(pp - 2)] / safe_scale;
                const double sk = s[static_cast<std::size_t>(k)] / safe_scale;
                const double ek = e[static_cast<std::size_t>(k)] / safe_scale;
                const double b_shift = 0.5
                    * ((spm1 + sp) * (spm1 - sp) + epm1 * epm1);
                const double c_shift = (sp * epm1) * (sp * epm1);

                double shift = 0.0;
                if (zero_shift_iterations < 4) {
                    ++zero_shift_iterations;
                } else if (exceptional_count >= 10) {
                    shift = 0.75
                        * std::abs(e[static_cast<std::size_t>(pp - 2)]) / safe_scale;
                    exceptional_count = 0;
                } else if (b_shift != 0.0 || c_shift != 0.0) {
                    const double discriminant = std::hypot(b_shift, std::sqrt(c_shift));
                    const double denominator = b_shift >= 0.0
                        ? b_shift + discriminant
                        : b_shift - discriminant;
                    if (denominator != 0.0 && std::isfinite(denominator)) {
                        shift = c_shift / denominator;
                    }
                }

                double f = (sk + sp) * (sk - sp) + shift;
                double g = sk * ek;

                for (int j = k; j < pp - 1; ++j) {
                    const GivensRotation right_rotation = make_givens(f, g);
                    if (!std::isfinite(right_rotation.r)) {
                        return JLC_STATUS_INTERNAL_ERROR;
                    }
                    if (j != k) {
                        e[static_cast<std::size_t>(j - 1)] = right_rotation.r;
                    }
                    f = right_rotation.c * s[static_cast<std::size_t>(j)]
                        + right_rotation.s * e[static_cast<std::size_t>(j)];
                    e[static_cast<std::size_t>(j)] =
                        right_rotation.c * e[static_cast<std::size_t>(j)]
                        - right_rotation.s * s[static_cast<std::size_t>(j)];
                    g = right_rotation.s * s[static_cast<std::size_t>(j + 1)];
                    s[static_cast<std::size_t>(j + 1)] =
                        right_rotation.c * s[static_cast<std::size_t>(j + 1)];
                    apply_givens_to_columns(right_factor, cols, cols, j, j + 1,
                                            right_rotation.c, right_rotation.s);

                    const GivensRotation left_rotation = make_givens(f, g);
                    if (!std::isfinite(left_rotation.r)) {
                        return JLC_STATUS_INTERNAL_ERROR;
                    }
                    s[static_cast<std::size_t>(j)] = left_rotation.r;
                    f = left_rotation.c * e[static_cast<std::size_t>(j)]
                        + left_rotation.s * s[static_cast<std::size_t>(j + 1)];
                    s[static_cast<std::size_t>(j + 1)] =
                        -left_rotation.s * e[static_cast<std::size_t>(j)]
                        + left_rotation.c * s[static_cast<std::size_t>(j + 1)];
                    g = left_rotation.s * e[static_cast<std::size_t>(j + 1)];
                    e[static_cast<std::size_t>(j + 1)] =
                        left_rotation.c * e[static_cast<std::size_t>(j + 1)];
                    apply_givens_to_columns(left_factor, rows, rows, j, j + 1,
                                            left_rotation.c, left_rotation.s);
                }
                e[static_cast<std::size_t>(pp - 2)] = f;
                break;
            }
            case 4: {
                if (s[static_cast<std::size_t>(k)] < 0.0) {
                    s[static_cast<std::size_t>(k)] =
                        -s[static_cast<std::size_t>(k)];
                    negate_column(right_factor, cols, cols, k);
                }
                while (k < pp - 1
                    && s[static_cast<std::size_t>(k)]
                        < s[static_cast<std::size_t>(k + 1)]) {
                    std::swap(s[static_cast<std::size_t>(k)],
                              s[static_cast<std::size_t>(k + 1)]);
                    swap_columns(right_factor, cols, cols, k, k + 1);
                    swap_columns(left_factor, rows, rows, k, k + 1);
                    ++k;
                }
                --pp;
                exceptional_count = 0;
                zero_shift_iterations = 0;
                break;
            }
            default:
                return JLC_STATUS_INTERNAL_ERROR;
        }

    }

    for (int i = 0; i < p; ++i) {
        if (s[static_cast<std::size_t>(i)] < 0.0) {
            s[static_cast<std::size_t>(i)] = -s[static_cast<std::size_t>(i)];
            negate_column(right_factor, cols, cols, i);
        }
        if (!std::isfinite(s[static_cast<std::size_t>(i)])) {
            return JLC_STATUS_INTERNAL_ERROR;
        }
    }

    // A final selection sort is cheap compared with the reduction and makes
    // the output contract explicit even when clustered values deflate together.
    for (int i = 0; i < p; ++i) {
        int largest = i;
        for (int j = i + 1; j < p; ++j) {
            if (s[static_cast<std::size_t>(j)]
                > s[static_cast<std::size_t>(largest)]) {
                largest = j;
            }
        }
        if (largest != i) {
            std::swap(s[static_cast<std::size_t>(i)],
                      s[static_cast<std::size_t>(largest)]);
            swap_columns(right_factor, cols, cols, i, largest);
            swap_columns(left_factor, rows, rows, i, largest);
        }
    }
    std::copy(s.begin(), s.end(), singular_values);
    return JLC_STATUS_SUCCESS;
}

jlc_status svd_bidiag_qr_scaled(const double* a, int m, int n,
                                double* u, double* singular_values, int singular_count,
                                double* v) {
    try {
        std::vector<double> normalized;
        double scale = 0.0;
        if (!normalize_input(a, m, n, normalized, scale)) {
            return JLC_STATUS_INVALID_ARGUMENT;
        }
        if (scale == 0.0) {
            set_identity(u, m);
            set_identity(v, n);
            std::fill(singular_values, singular_values + singular_count, 0.0);
            return JLC_STATUS_SUCCESS;
        }

        if (m >= n) {
            const std::size_t b_size = static_cast<std::size_t>(m)
                * static_cast<std::size_t>(n);
            const std::size_t u_size = static_cast<std::size_t>(m)
                * static_cast<std::size_t>(m);
            const std::size_t v_size = static_cast<std::size_t>(n)
                * static_cast<std::size_t>(n);
            std::vector<double> b(b_size, 0.0);
            std::vector<double> u_factor(u_size, 0.0);
            std::vector<double> v_factor(v_size, 0.0);
            jlc_status status = jlc_native_bidiagonal_decompose(
                normalized.data(), m, n, u_factor.data(), b.data(), v_factor.data());
            if (status != JLC_STATUS_SUCCESS) {
                return status;
            }
            status = bidiagonal_qr(b.data(), m, n, u_factor.data(), v_factor.data(),
                                   singular_values);
            if (status != JLC_STATUS_SUCCESS) {
                return status;
            }
            if (!rescale_singular_values(singular_values, singular_count, scale)) {
                return JLC_STATUS_INTERNAL_ERROR;
            }
            std::copy(u_factor.begin(), u_factor.end(), u);
            std::copy(v_factor.begin(), v_factor.end(), v);
            return JLC_STATUS_SUCCESS;
        }

        // Work on A^T so the shared native bidiagonalizer always receives the
        // upper-bidiagonal orientation. Exchange the accumulated factors back
        // at the end: A = V_t S U_t^T.
        const std::size_t transposed_size = static_cast<std::size_t>(n)
            * static_cast<std::size_t>(m);
        std::vector<double> transposed(transposed_size, 0.0);
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                transposed[static_cast<std::size_t>(col) * m + row] =
                    normalized[static_cast<std::size_t>(row) * n + col];
            }
        }
        const std::size_t u_t_size = static_cast<std::size_t>(n)
            * static_cast<std::size_t>(n);
        const std::size_t v_t_size = static_cast<std::size_t>(m)
            * static_cast<std::size_t>(m);
        std::vector<double> b_t(transposed_size, 0.0);
        std::vector<double> u_t(u_t_size, 0.0);
        std::vector<double> v_t(v_t_size, 0.0);
        jlc_status status = jlc_native_bidiagonal_decompose(
            transposed.data(), n, m, u_t.data(), b_t.data(), v_t.data());
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
        status = bidiagonal_qr(b_t.data(), n, m, u_t.data(), v_t.data(), singular_values);
        if (status != JLC_STATUS_SUCCESS) {
            return status;
        }
        if (!rescale_singular_values(singular_values, singular_count, scale)) {
            return JLC_STATUS_INTERNAL_ERROR;
        }
        std::copy(v_t.begin(), v_t.end(), u);
        std::copy(u_t.begin(), u_t.end(), v);
        return JLC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return JLC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return JLC_STATUS_INTERNAL_ERROR;
    }
}

bool valid_svd_outputs(const double* u, const double* singular_values, const double* v,
                       int m, int n, int rank) {
    const std::size_t u_size = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
    const std::size_t v_size = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    for (std::size_t i = 0; i < u_size; ++i) {
        if (!std::isfinite(u[i])) {
            return false;
        }
    }
    for (std::size_t i = 0; i < v_size; ++i) {
        if (!std::isfinite(v[i])) {
            return false;
        }
    }
    double previous = std::numeric_limits<double>::infinity();
    for (int i = 0; i < rank; ++i) {
        const double value = singular_values[i];
        if (!std::isfinite(value) || value < 0.0) {
            return false;
        }
        if (value > previous
            && value - previous > QR_TOL * std::max(1.0, std::max(value, previous))) {
            return false;
        }
        previous = value;
    }
    return true;
}

jlc_svd_algorithm select_svd_algorithm(int m, int n) {
    // The Java facade makes the normal dispatch decision. C callers using
    // AUTO get the production path as well; Jacobi remains explicitly
    // selectable for small-matrix comparisons and platform-specific tuning.
    (void)m;
    (void)n;
    return JLC_SVD_ALGORITHM_BIDIAG_QR;
}

} // namespace

jlc_status jlc_native_svd_decompose(const double* a, int m, int n,
                                    double* u, double* singular_values, int singular_count,
                                    double* v) {
    return jlc_native_svd_decompose_with_algorithm(
        a, m, n, u, singular_values, singular_count, v, JLC_SVD_ALGORITHM_AUTO);
}

jlc_status jlc_native_svd_decompose_with_algorithm(const double* a, int m, int n,
                                                   double* u, double* singular_values,
                                                   int singular_count, double* v,
                                                   int algorithm) {
    if (a == nullptr || u == nullptr || singular_values == nullptr || v == nullptr
        || m < 0 || n < 0 || singular_count != std::min(m, n)) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (algorithm != JLC_SVD_ALGORITHM_AUTO
        && algorithm != JLC_SVD_ALGORITHM_JACOBI
        && algorithm != JLC_SVD_ALGORITHM_BIDIAG_QR) {
        return JLC_STATUS_INVALID_ARGUMENT;
    }
    if (m == 0 || n == 0) {
        set_identity(u, m);
        set_identity(v, n);
        return JLC_STATUS_SUCCESS;
    }

    const jlc_svd_algorithm selected = algorithm == JLC_SVD_ALGORITHM_AUTO
        ? select_svd_algorithm(m, n)
        : static_cast<jlc_svd_algorithm>(algorithm);
    jlc_status status = selected == JLC_SVD_ALGORITHM_JACOBI
        ? svd_jacobi_scaled(a, m, n, u, singular_values, singular_count, v)
        : svd_bidiag_qr_scaled(a, m, n, u, singular_values, singular_count, v);
    if (status != JLC_STATUS_SUCCESS) {
        return status;
    }
    return valid_svd_outputs(u, singular_values, v, m, n, singular_count)
        ? JLC_STATUS_SUCCESS : JLC_STATUS_INTERNAL_ERROR;
}
