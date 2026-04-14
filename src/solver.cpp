#include "super_ik_tsinghua_1dqp/solver.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace super_ik_tsinghua_1dqp {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxWeight = 1e6;

struct Interval {
  double start {};
  double end {};
};

struct BranchSamples {
  std::vector<double> psi;
  std::vector<Vec7> q;
};

struct CosPsiModel {
  double a {0.0};
  double b {0.0};
  double c {0.0};
  bool ok {false};
};

struct AtanPsiModel {
  double an {0.0};
  double bn {0.0};
  double cn {0.0};
  double ad {0.0};
  double bd {0.0};
  double cd {0.0};
  bool ok {false};
};

struct LinearizedModel {
  bool ok {false};
  double psi0 {0.0};
  Vec7 q0 {Vec7::Zero()};
  Vec7 k {Vec7::Zero()};
};

double WrapToPi(double x) {
  double y = std::fmod(x + kPi, 2.0 * kPi);
  if (y < 0.0) y += 2.0 * kPi;
  return y - kPi;
}

Vec7 WrapToPi(const Vec7& q) {
  Vec7 out;
  for (int i = 0; i < 7; ++i) out[i] = WrapToPi(q[i]);
  return out;
}

Eigen::Matrix3d RotX(double t) {
  const double c = std::cos(t);
  const double s = std::sin(t);
  Eigen::Matrix3d r;
  r << 1.0, 0.0, 0.0,
       0.0, c, -s,
       0.0, s, c;
  return r;
}

Eigen::Matrix3d RotZ(double t) {
  const double c = std::cos(t);
  const double s = std::sin(t);
  Eigen::Matrix3d r;
  r << c, -s, 0.0,
       s, c, 0.0,
       0.0, 0.0, 1.0;
  return r;
}

Mat4 DhA(double theta, double d, double a_prev, double alpha_prev) {
  Mat4 t = Mat4::Identity();
  t.block<3, 3>(0, 0) = RotX(alpha_prev) * RotZ(theta);
  t.block<3, 1>(0, 3) = Eigen::Vector3d(
      a_prev, -std::sin(alpha_prev) * d, std::cos(alpha_prev) * d);
  return t;
}

double PoseErrorNorm(const Mat4& cur, const Mat4& des) {
  const Eigen::Vector3d dp = cur.block<3, 1>(0, 3) - des.block<3, 1>(0, 3);
  const Eigen::Matrix3d r = cur.block<3, 3>(0, 0);
  const Eigen::Matrix3d rd = des.block<3, 3>(0, 0);
  const Eigen::Vector3d re = 0.5 * (
      r.col(0).cross(rd.col(0)) +
      r.col(1).cross(rd.col(1)) +
      r.col(2).cross(rd.col(2)));
  Eigen::Matrix<double, 6, 1> e;
  e << dp, re;
  return e.norm();
}

bool WithinLimits(const Vec7& q, const Eigen::Matrix<double, 7, 2>& limits) {
  for (int i = 0; i < 7; ++i) {
    if (q[i] < limits(i, 0) - 1e-8 || q[i] > limits(i, 1) + 1e-8) return false;
  }
  return true;
}

void ComputeSweFromTarget(
    const Mat4& t07, const NeroParams& p,
    Eigen::Vector3d* s, Eigen::Vector3d* w, double* q4_abs) {
  const Eigen::Matrix3d r = t07.block<3, 3>(0, 0);
  const Eigen::Vector3d p_target = t07.block<3, 1>(0, 3);
  const Eigen::Vector3d z7 = r.col(2);
  const Eigen::Vector3d o7 = p_target - p.post_transform_d8 * z7;
  *w = o7 - p.d_i[6] * z7;
  *s = Eigen::Vector3d(0.0, 0.0, p.d_i[0]);

  const double l_sw = (*w - *s).norm();
  const double l_se = std::abs(p.d_i[2]);
  const double l_ew = std::abs(p.d_i[4]);
  if (l_sw < 1e-10) {
    *q4_abs = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  const double c4 = (l_sw * l_sw - l_se * l_se - l_ew * l_ew) / (2.0 * l_se * l_ew);
  if (c4 < -1.0 || c4 > 1.0) {
    *q4_abs = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  *q4_abs = std::acos(std::clamp(c4, -1.0, 1.0));
}

std::optional<Eigen::Vector3d> ElbowFromArmAngle(
    const Eigen::Vector3d& s,
    const Eigen::Vector3d& w,
    double psi,
    const NeroParams& p) {
  const double l_se = std::abs(p.d_i[2]);
  const double l_ew = std::abs(p.d_i[4]);
  const Eigen::Vector3d sw = w - s;
  const double l_sw = sw.norm();
  if (l_sw < 1e-12) return std::nullopt;
  const Eigen::Vector3d u_sw = sw / l_sw;
  const double x = (l_se * l_se - l_ew * l_ew + l_sw * l_sw) / (2.0 * l_sw);
  const double r2 = l_se * l_se - x * x;
  if (r2 < -1e-10) return std::nullopt;
  const double rc = std::sqrt(std::max(0.0, r2));
  const Eigen::Vector3d c = s + x * u_sw;

  Eigen::Vector3d t = s.cross(u_sw);
  if (t.norm() < 1e-10) t = Eigen::Vector3d(1.0, 0.0, 0.0).cross(u_sw);
  if (t.norm() < 1e-10) t = Eigen::Vector3d(0.0, 1.0, 0.0).cross(u_sw);
  const Eigen::Vector3d e1 = t.normalized();
  const Eigen::Vector3d e2 = u_sw.cross(e1).normalized();
  return c + rc * (std::cos(psi) * e1 + std::sin(psi) * e2);
}

std::vector<Vec7> SolveQ123FromSwe(
    const Eigen::Vector3d& e,
    const Eigen::Vector3d& w,
    double q4,
    const NeroParams& p) {
  std::vector<Vec7> out;
  const double d0 = p.d_i[0];
  const double d2 = p.d_i[2];
  const double d4 = p.d_i[4];
  if (std::abs(d2) < 1e-12 || std::abs(d4) < 1e-12) return out;

  const double ex = e.x();
  const double ey = e.y();
  const double ez = e.z();
  const double rho = std::hypot(ex, ey);
  double c2 = (ez - d0) / d2;
  if (c2 < -1.0 - 1e-8 || c2 > 1.0 + 1e-8) return out;
  c2 = std::clamp(c2, -1.0, 1.0);
  const double s2_abs = std::sqrt(std::max(0.0, 1.0 - c2 * c2));
  if (rho > std::abs(d2) + 1e-7) return out;

  Eigen::Vector3d col2 = -(w - e) / d4;
  if (col2.norm() < 1e-10) return out;
  col2.normalize();
  const double u1 = col2.x();
  const double u2 = col2.y();
  const double u3 = col2.z();
  const double s4 = std::sin(q4);
  const double c4 = std::cos(q4);
  if (std::abs(s4) < 1e-8) return out;

  for (double s2 : {s2_abs, -s2_abs}) {
    if (std::abs(s2) < 1e-10) continue;
    double c1 = -ex / (d2 * s2);
    double s1 = -ey / (d2 * s2);
    const double n1 = std::hypot(c1, s1);
    if (n1 < 1e-12) continue;
    c1 /= n1;
    s1 /= n1;
    const double q1 = std::atan2(s1, c1);
    const double q2 = std::atan2(s2, c2);

    const double b1 = (s2 * c1 * c4 - u1) / s4;
    const double b2 = (u2 - s1 * s2 * c4) / s4;
    double s3 = s1 * b1 + c1 * b2;
    double c3 = (std::abs(c2) > 1e-8) ? ((-c1 * b1 + s1 * b2) / c2) : ((u3 + c2 * c4) / (s2 * s4));
    const double n3 = std::hypot(s3, c3);
    if (n3 < 1e-12) continue;
    s3 /= n3;
    c3 /= n3;
    Vec7 q = Vec7::Zero();
    q[0] = q1;
    q[1] = q2;
    q[2] = std::atan2(s3, c3);
    out.push_back(q);
  }
  return out;
}

std::vector<Eigen::Vector3d> Extract567(const Mat4& t47) {
  std::vector<Eigen::Vector3d> out;
  const double c6 = std::clamp(t47(1, 2), -1.0, 1.0);
  for (double sgn : {1.0, -1.0}) {
    const double s6 = sgn * std::sqrt(std::max(0.0, 1.0 - c6 * c6));
    if (std::abs(s6) < 1e-8) continue;
    out.emplace_back(
        std::atan2(t47(2, 2) / s6, t47(0, 2) / s6),
        std::atan2(s6, c6),
        std::atan2(t47(1, 1) / s6, -t47(1, 0) / s6));
  }
  return out;
}

Vec7 NormalizeJointPosition(const Vec7& q, const NeroParams& p) {
  Vec7 x = Vec7::Zero();
  for (int i = 0; i < 7; ++i) {
    const double ql = p.joint_limits(i, 0);
    const double qu = p.joint_limits(i, 1);
    x[i] = 2.0 * (q[i] - (qu + ql) * 0.5) / (qu - ql);
  }
  return x;
}

double WeightFromNormalized(double x) {
  constexpr double a = 2.38;
  constexpr double b = 2.28;
  if (x >= 0.0) {
    if (x >= 1.0) return kMaxWeight;
    return std::min(kMaxWeight, b * x / (std::exp(a * (1.0 - x)) - 1.0));
  }
  if (x <= -1.0) return kMaxWeight;
  return std::min(kMaxWeight, -b * x / (std::exp(a * (1.0 + x)) - 1.0));
}

Vec7 WeightLimitsFromPrev(const Vec7& q_prev, const NeroParams& p) {
  const Vec7 x_prev = NormalizeJointPosition(q_prev, p);
  Vec7 w = Vec7::Zero();
  for (int i = 0; i < 7; ++i) w[i] = WeightFromNormalized(x_prev[i]);
  return w;
}

std::vector<int> DangerJointIndices(const Vec7& q_prev, const NeroParams& p, double danger_threshold) {
  std::vector<int> idx;
  const Vec7 x_prev = NormalizeJointPosition(q_prev, p);
  for (int i = 0; i < 7; ++i) {
    if (std::abs(x_prev[i]) >= danger_threshold) idx.push_back(i);
  }
  return idx;
}

double DistanceToReference(const Vec7& q, const Vec7& q_ref) {
  return WrapToPi(q - q_ref).norm();
}

std::vector<Vec7> IkOneArmAngleInternal(const Mat4& target, double psi, const NeroParams& p) {
  std::vector<Vec7> sols;
  Eigen::Vector3d s, w;
  double q4_abs = 0.0;
  ComputeSweFromTarget(target, p, &s, &w, &q4_abs);
  if (!std::isfinite(q4_abs)) return sols;
  const auto e = ElbowFromArmAngle(s, w, psi, p);
  if (!e.has_value()) return sols;

  Mat4 t_post = Mat4::Identity();
  t_post(2, 3) = p.post_transform_d8;
  const Mat4 t_chain = target * t_post.inverse();

  for (double q4 : {q4_abs, -q4_abs}) {
    auto q123_sols = SolveQ123FromSwe(*e, w, q4, p);
    for (const auto& q123 : q123_sols) {
      const double th1 = q123[0] + p.theta_offset[0];
      const double th2 = q123[1] + p.theta_offset[1];
      const double th3 = q123[2] + p.theta_offset[2];
      const double th4 = q4 + p.theta_offset[3];
      Mat4 t04 = Mat4::Identity();
      const double th_raw[4] = {th1, th2, th3, th4};
      for (int i = 0; i < 4; ++i) t04 = t04 * DhA(th_raw[i], p.d_i[i], p.a_prev[i], p.alpha_prev[i]);
      const auto q567_sols = Extract567(t04.inverse() * t_chain);
      for (const auto& q567 : q567_sols) {
        Vec7 theta_raw;
        theta_raw << th1, th2, th3, th4, q567[0], q567[1], q567[2];
        Vec7 q = WrapToPi(theta_raw - p.theta_offset);
        if (WithinLimits(q, p.joint_limits)) sols.push_back(q);
      }
    }
  }
  return sols;
}

std::optional<Vec7> SelectBranchNearest(const Mat4& target, double psi, const Vec7& q_ref, const NeroParams& p) {
  const auto sols = IkOneArmAngleInternal(target, psi, p);
  if (sols.empty()) return std::nullopt;
  int best_idx = 0;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int i = 0; i < static_cast<int>(sols.size()); ++i) {
    const double cost = DistanceToReference(sols[i], q_ref);
    if (cost < best_cost) {
      best_cost = cost;
      best_idx = i;
    }
  }
  return sols[best_idx];
}

std::optional<std::pair<double, Vec7>> FindNearestFeasibleSeed(
    const Mat4& target,
    double psi_ref,
    const Vec7& q_ref,
    const NeroParams& p,
    const PaperParams& paper) {
  auto q = SelectBranchNearest(target, psi_ref, q_ref, p);
  if (q.has_value()) return std::make_pair(psi_ref, *q);

  std::optional<std::pair<double, Vec7>> best;
  double best_cost = std::numeric_limits<double>::infinity();
  const int kSamples = std::max(9, paper.psi_seed_samples);
  for (int i = 0; i < kSamples; ++i) {
    const double psi = -kPi + (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(kSamples - 1);
    auto cand = SelectBranchNearest(target, psi, q_ref, p);
    if (!cand.has_value()) continue;
    const double cost = std::abs(WrapToPi(psi - psi_ref)) + 0.1 * DistanceToReference(*cand, q_ref);
    if (cost < best_cost) {
      best_cost = cost;
      best = std::make_pair(psi, *cand);
    }
  }
  return best;
}

std::vector<double> BuildPsiGrid(double step) {
  std::vector<double> psi;
  for (double p = -kPi; p <= kPi + 1e-12; p += step) psi.push_back(std::min(kPi, p));
  if (psi.empty() || std::abs(psi.back() - kPi) > 1e-9) psi.push_back(kPi);
  return psi;
}

BranchSamples TrackContinuousBranchSamples(
    const Mat4& target,
    double psi_ref,
    const Vec7& q_ref,
    const NeroParams& p,
    const PaperParams& paper) {
  BranchSamples out;
  const auto grid = BuildPsiGrid(paper.feasible_scan_step);
  if (grid.empty()) return out;
  int ref_idx = 0;
  double best_dist = std::numeric_limits<double>::infinity();
  for (int i = 0; i < static_cast<int>(grid.size()); ++i) {
    const double d = std::abs(grid[i] - psi_ref);
    if (d < best_dist) {
      best_dist = d;
      ref_idx = i;
    }
  }
  auto q0 = SelectBranchNearest(target, grid[ref_idx], q_ref, p);
  if (!q0.has_value()) return out;
  out.psi.push_back(grid[ref_idx]);
  out.q.push_back(*q0);

  Vec7 ref = *q0;
  for (int i = ref_idx + 1; i < static_cast<int>(grid.size()); ++i) {
    auto q = SelectBranchNearest(target, grid[i], ref, p);
    if (!q.has_value()) break;
    out.psi.push_back(grid[i]);
    out.q.push_back(*q);
    ref = *q;
  }

  ref = *q0;
  std::vector<double> rev_psi;
  std::vector<Vec7> rev_q;
  for (int i = ref_idx - 1; i >= 0; --i) {
    auto q = SelectBranchNearest(target, grid[i], ref, p);
    if (!q.has_value()) break;
    rev_psi.push_back(grid[i]);
    rev_q.push_back(*q);
    ref = *q;
  }
  for (int i = static_cast<int>(rev_psi.size()) - 1; i >= 0; --i) {
    out.psi.insert(out.psi.begin(), rev_psi[i]);
    out.q.insert(out.q.begin(), rev_q[i]);
  }
  return out;
}

bool IsCosJointIndex(int idx) {
  return idx == 1 || idx == 5;
}

CosPsiModel FitCosPsiModel(const BranchSamples& samples, int joint_idx) {
  CosPsiModel model;
  if (samples.psi.size() < 3) return model;
  Eigen::MatrixXd a(samples.psi.size(), 3);
  Eigen::VectorXd b(samples.psi.size());
  for (int i = 0; i < static_cast<int>(samples.psi.size()); ++i) {
    a(i, 0) = std::sin(samples.psi[i]);
    a(i, 1) = std::cos(samples.psi[i]);
    a(i, 2) = 1.0;
    b(i) = std::cos(samples.q[i][joint_idx]);
  }
  const Eigen::Vector3d x = a.colPivHouseholderQr().solve(b);
  model.a = x[0];
  model.b = x[1];
  model.c = x[2];
  model.ok = true;
  return model;
}

AtanPsiModel FitAtanPsiModel(const BranchSamples& samples, int joint_idx) {
  AtanPsiModel model;
  if (samples.psi.size() < 6) return model;
  Eigen::MatrixXd a(2 * samples.psi.size(), 6);
  Eigen::VectorXd b(2 * samples.psi.size());
  for (int i = 0; i < static_cast<int>(samples.psi.size()); ++i) {
    const double psi = samples.psi[i];
    const double q = samples.q[i][joint_idx];
    a.row(2 * i) << std::sin(psi), std::cos(psi), 1.0, 0.0, 0.0, 0.0;
    a.row(2 * i + 1) << 0.0, 0.0, 0.0, std::sin(psi), std::cos(psi), 1.0;
    b[2 * i] = std::sin(q);
    b[2 * i + 1] = std::cos(q);
  }
  const Eigen::Matrix<double, 6, 1> x = a.colPivHouseholderQr().solve(b);
  model.an = x[0];
  model.bn = x[1];
  model.cn = x[2];
  model.ad = x[3];
  model.bd = x[4];
  model.cd = x[5];
  model.ok = true;
  return model;
}

std::vector<double> SolveTrigEquation(double a, double b, double c) {
  std::vector<double> roots;
  const double r = std::hypot(a, b);
  if (r < 1e-12) return roots;
  const double rhs = -c / r;
  if (rhs < -1.0 - 1e-9 || rhs > 1.0 + 1e-9) return roots;
  const double phi = std::atan2(b, a);
  const double y = std::asin(std::clamp(rhs, -1.0, 1.0));
  roots.push_back(WrapToPi(y - phi));
  roots.push_back(WrapToPi((kPi - y) - phi));
  std::sort(roots.begin(), roots.end());
  roots.erase(std::unique(roots.begin(), roots.end(), [](double x, double y) {
    return std::abs(x - y) < 1e-8;
  }), roots.end());
  return roots;
}

std::vector<Interval> IntersectIntervals(const std::vector<Interval>& a, const std::vector<Interval>& b) {
  if (a.empty() || b.empty()) return {};
  std::vector<Interval> out;
  for (const auto& ia : a) {
    for (const auto& ib : b) {
      const double s = std::max(ia.start, ib.start);
      const double e = std::min(ia.end, ib.end);
      if (s <= e + 1e-10) out.push_back({s, e});
    }
  }
  return out;
}

std::vector<Interval> MergeIntervals(std::vector<Interval> intervals, double gap_tol = 1e-4) {
  if (intervals.empty()) return intervals;
  std::sort(intervals.begin(), intervals.end(), [](const Interval& lhs, const Interval& rhs) {
    if (lhs.start == rhs.start) return lhs.end < rhs.end;
    return lhs.start < rhs.start;
  });
  std::vector<Interval> merged;
  merged.push_back(intervals.front());
  for (int i = 1; i < static_cast<int>(intervals.size()); ++i) {
    auto& back = merged.back();
    if (intervals[i].start <= back.end + gap_tol) {
      back.end = std::max(back.end, intervals[i].end);
    } else {
      merged.push_back(intervals[i]);
    }
  }
  return merged;
}

std::vector<Interval> SampleIntervals(
    const std::function<double(double)>& eval,
    double q_lo,
    double q_hi,
    const std::vector<double>& extra_cuts = {}) {
  std::vector<double> cuts = {-kPi, kPi};
  cuts.insert(cuts.end(), extra_cuts.begin(), extra_cuts.end());
  const int dense_n = 720;
  for (int i = 0; i <= dense_n; ++i) {
    const double psi = -kPi + (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(dense_n);
    const double q = eval(psi);
    if (std::abs(WrapToPi(q - q_lo)) < 0.02 || std::abs(WrapToPi(q - q_hi)) < 0.02) cuts.push_back(psi);
  }
  std::sort(cuts.begin(), cuts.end());
  cuts.erase(std::unique(cuts.begin(), cuts.end(), [](double x, double y) {
    return std::abs(x - y) < 1e-5;
  }), cuts.end());
  std::vector<Interval> out;
  for (int i = 0; i + 1 < static_cast<int>(cuts.size()); ++i) {
    const double s = cuts[i];
    const double e = cuts[i + 1];
    const double mid = 0.5 * (s + e);
    const double q = eval(mid);
    if (q >= q_lo - 1e-7 && q <= q_hi + 1e-7) out.push_back({s, e});
  }
  return out;
}

std::vector<Interval> BuildCosJointIntervals(const CosPsiModel& model, double q_lo, double q_hi) {
  auto eval = [&](double psi) {
    const double c = model.a * std::sin(psi) + model.b * std::cos(psi) + model.c;
    return std::acos(std::clamp(c, -1.0, 1.0));
  };
  return SampleIntervals(eval, q_lo, q_hi);
}

std::vector<Interval> BuildAtanJointIntervals(const AtanPsiModel& model, double q_lo, double q_hi) {
  const double at = model.cn * model.bd - model.bn * model.cd;
  const double bt = model.an * model.cd - model.cn * model.ad;
  const double ct = model.an * model.bd - model.bn * model.ad;
  auto eval = [&](double psi) {
    const double u = model.an * std::sin(psi) + model.bn * std::cos(psi) + model.cn;
    const double v = model.ad * std::sin(psi) + model.bd * std::cos(psi) + model.cd;
    return WrapToPi(std::atan2(u, v));
  };
  return SampleIntervals(eval, q_lo, q_hi, SolveTrigEquation(at, bt, ct));
}

std::vector<Interval> BuildDangerJointFeasibleIntervals(
    const Mat4& target,
    const Vec7& q_prev,
    double psi_prev,
    const NeroParams& p,
    const PaperParams& paper) {
  auto seed = FindNearestFeasibleSeed(target, psi_prev, q_prev, p, paper);
  if (!seed.has_value()) return {};
  const auto danger_joint_idx = DangerJointIndices(q_prev, p, paper.danger_threshold);
  if (danger_joint_idx.empty()) return {{-kPi, kPi}};
  const BranchSamples samples = TrackContinuousBranchSamples(target, seed->first, seed->second, p, paper);
  if (samples.psi.size() < 6) return {};

  std::vector<Interval> feasible {{-kPi, kPi}};
  for (int joint_idx : danger_joint_idx) {
    std::vector<Interval> joint_intervals;
    if (IsCosJointIndex(joint_idx)) {
      const auto model = FitCosPsiModel(samples, joint_idx);
      if (!model.ok) return {};
      joint_intervals = BuildCosJointIntervals(model, p.joint_limits(joint_idx, 0), p.joint_limits(joint_idx, 1));
    } else {
      const auto model = FitAtanPsiModel(samples, joint_idx);
      if (!model.ok) return {};
      joint_intervals = BuildAtanJointIntervals(model, p.joint_limits(joint_idx, 0), p.joint_limits(joint_idx, 1));
    }
    feasible = IntersectIntervals(feasible, joint_intervals);
    if (feasible.empty()) return {};
  }
  return MergeIntervals(std::move(feasible));
}

double ProjectPsiToFeasibleSet(double psi, const std::vector<Interval>& feasible) {
  for (const auto& seg : feasible) {
    if (psi >= seg.start && psi <= seg.end) return psi;
  }
  double best = feasible.front().start;
  double best_dist = std::numeric_limits<double>::infinity();
  for (const auto& seg : feasible) {
    for (double cand : {seg.start, seg.end}) {
      const double d = std::abs(WrapToPi(psi - cand));
      if (d < best_dist) {
        best_dist = d;
        best = cand;
      }
    }
  }
  return best;
}

bool PsiInsideIntervals(double psi, const std::vector<Interval>& feasible) {
  for (const auto& seg : feasible) {
    if (psi >= seg.start - 1e-10 && psi <= seg.end + 1e-10) return true;
  }
  return false;
}

double WeightedJointCostFromWeights(const Vec7& q, const Vec7& q_prev, const Vec7& w) {
  const Vec7 dq = WrapToPi(q - q_prev);
  double cost = 0.0;
  for (int i = 0; i < 7; ++i) cost += w[i] * dq[i] * dq[i];
  return cost;
}

std::optional<std::pair<double, Vec7>> RecoverBranchNearPsi(
    const Mat4& target,
    const Vec7& q_ref,
    double psi_anchor,
    const std::vector<Interval>& feasible,
    const PaperParams& paper,
    const NeroParams& p) {
  const int n_local = std::max(9, paper.psi_recover_samples);
  std::optional<std::pair<double, Vec7>> best;
  double best_cost = std::numeric_limits<double>::infinity();
  for (const auto& seg : feasible) {
    for (int i = 0; i < n_local; ++i) {
      const double t = (n_local == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(n_local - 1);
      const double psi = seg.start + t * (seg.end - seg.start);
      if (std::abs(WrapToPi(psi - psi_anchor)) > paper.psi_recover_window) continue;
      auto cand = SelectBranchNearest(target, psi, q_ref, p);
      if (!cand.has_value()) continue;
      const double cost = std::abs(WrapToPi(psi - psi_anchor)) + 0.05 * DistanceToReference(*cand, q_ref);
      if (cost < best_cost) {
        best_cost = cost;
        best = std::make_pair(psi, *cand);
      }
    }
  }
  return best;
}

std::optional<std::pair<double, Vec7>> GlobalFallbackCandidate(
    const Mat4& target,
    const Vec7& q_prev,
    const Vec7& q_ref,
    const Vec7& w,
    const std::vector<Interval>& feasible,
    const PaperParams& paper,
    const NeroParams& p,
    bool restrict_to_feasible) {
  const int n = std::max(31, paper.global_fallback_samples);
  std::optional<std::pair<double, Vec7>> best;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    const double psi = -kPi + (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(n - 1);
    if (restrict_to_feasible && !feasible.empty() && !PsiInsideIntervals(psi, feasible)) continue;
    auto cand = SelectBranchNearest(target, psi, q_ref, p);
    if (!cand.has_value()) continue;
    const double cost = WeightedJointCostFromWeights(*cand, q_prev, w);
    if (cost < best_cost) {
      best_cost = cost;
      best = std::make_pair(psi, *cand);
    }
  }
  return best;
}

LinearizedModel LinearizeJointAroundPsi(
    const Mat4& target,
    const Vec7& q_prev,
    double psi_prev,
    const NeroParams& p,
    const PaperParams& paper) {
  LinearizedModel out;
  auto seed = FindNearestFeasibleSeed(target, psi_prev, q_prev, p, paper);
  if (!seed.has_value()) return out;
  const double psi0 = seed->first;
  const Vec7 q0 = seed->second;
  auto q1 = SelectBranchNearest(target, WrapToPi(psi0 + paper.delta_psi), q0, p);
  if (!q1.has_value()) return out;
  out.ok = true;
  out.psi0 = psi0;
  out.q0 = q0;
  out.k = WrapToPi(*q1 - q0) / paper.delta_psi;
  return out;
}

std::vector<std::pair<double, double>> ToPairIntervals(const std::vector<Interval>& intervals) {
  std::vector<std::pair<double, double>> out;
  for (const auto& seg : intervals) out.emplace_back(seg.start, seg.end);
  return out;
}

}  // namespace

NeroParams NeroParams::Default() {
  NeroParams p;
  p.a_prev << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  p.alpha_prev << 0.0, kPi / 2.0, kPi / 2.0, kPi / 2.0, kPi / 2.0, kPi / 2.0, kPi / 2.0;
  p.d_i << 0.138, 0.0, 0.31, 0.0, 0.27, 0.0, 0.0235;
  p.theta_offset << 0.0, -kPi, -kPi, -kPi, kPi / 2.0, kPi / 2.0, 0.0;
  p.joint_limits <<
      -2.705261, 2.705261,
      -1.745330, 1.745330,
      -2.757621, 2.757621,
      -1.012291, 2.146755,
      -2.757621, 2.757621,
      -0.733039, 0.959932,
      -1.570797, 1.570797;
  // Default IK target is link7 origin (no virtual TCP offset).
  p.post_transform_d8 = 0.0;
  return p;
}

Solver::Solver(NeroParams params, PaperParams paper)
    : params_(std::move(params)), paper_(paper) {}

std::vector<Mat4> Solver::FkAll(const Vec7& q) const {
  std::vector<Mat4> out;
  out.reserve(8);
  Mat4 t = Mat4::Identity();
  out.push_back(t);
  for (int i = 0; i < 7; ++i) {
    t = t * DhA(q[i] + params_.theta_offset[i], params_.d_i[i], params_.a_prev[i], params_.alpha_prev[i]);
    out.push_back(t);
  }
  return out;
}

Mat4 Solver::Fk(const Vec7& q) const {
  Mat4 t = FkAll(q).back();
  Mat4 t_post = Mat4::Identity();
  t_post(2, 3) = params_.post_transform_d8;
  return t * t_post;
}

IkResult Solver::IkArmAngle(const Mat4& target, const Vec7& q_prev, double psi_prev) const {
  IkResult res;
  res.report.method = "paper_1d_qp";
  res.report.psi_prev = psi_prev;
  res.report.delta_psi = paper_.delta_psi;
  res.report.danger_joint_indices = DangerJointIndices(q_prev, params_, paper_.danger_threshold);
  const Vec7 w = WeightLimitsFromPrev(q_prev, params_);

  const auto feasible = BuildDangerJointFeasibleIntervals(target, q_prev, psi_prev, params_, paper_);
  res.report.feasible_intervals = ToPairIntervals(feasible);
  if (feasible.empty()) {
    if (paper_.enable_global_fallback) {
      auto seed = FindNearestFeasibleSeed(target, psi_prev, q_prev, params_, paper_);
      const Vec7 q_ref = seed.has_value() ? seed->second : q_prev;
      auto global = GlobalFallbackCandidate(
          target, q_prev, q_ref, w, feasible, paper_, params_, false);
      if (global.has_value()) {
        res.report.fallback_used = true;
        res.report.psi_selected = global->first;
        res.q_best = global->second;
        res.report.pose_err_best = PoseErrorNorm(Fk(*res.q_best), target);
        res.report.success = true;
      }
    }
    return res;
  }

  const auto linearized = LinearizeJointAroundPsi(target, q_prev, psi_prev, params_, paper_);
  if (!linearized.ok) return res;

  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  for (int i = 0; i < 7; ++i) {
    const double d = linearized.q0[i] - q_prev[i] - linearized.k[i] * linearized.psi0;
    a += w[i] * linearized.k[i] * linearized.k[i];
    b += 2.0 * w[i] * linearized.k[i] * d;
    c += w[i] * d * d;
  }
  res.report.quad_A = a;
  res.report.quad_B = b;
  res.report.quad_C = c;

  double psi_opt = psi_prev;
  if (std::abs(a) > 1e-12) psi_opt = -b / (2.0 * a);
  psi_opt = WrapToPi(psi_opt);
  res.report.psi_opt = psi_opt;
  const double psi_final = ProjectPsiToFeasibleSet(psi_opt, feasible);
  res.report.psi_selected = psi_final;

  auto q_best = SelectBranchNearest(target, psi_final, linearized.q0, params_);
  if (!q_best.has_value()) {
    auto recovered = RecoverBranchNearPsi(
        target, linearized.q0, psi_final, feasible, paper_, params_);
    if (recovered.has_value()) {
      res.report.fallback_used = true;
      res.report.psi_selected = recovered->first;
      q_best = recovered->second;
    }
  }
  if (!q_best.has_value() && paper_.enable_global_fallback) {
    auto global = GlobalFallbackCandidate(
        target, q_prev, linearized.q0, w, feasible, paper_, params_, true);
    if (global.has_value()) {
      res.report.fallback_used = true;
      res.report.psi_selected = global->first;
      q_best = global->second;
    }
  }
  if (!q_best.has_value() && paper_.enable_global_fallback) {
    auto global = GlobalFallbackCandidate(
        target, q_prev, linearized.q0, w, feasible, paper_, params_, false);
    if (global.has_value()) {
      res.report.fallback_used = true;
      res.report.psi_selected = global->first;
      q_best = global->second;
    }
  }
  if (!q_best.has_value()) return res;
  res.q_best = *q_best;
  res.report.pose_err_best = PoseErrorNorm(Fk(*q_best), target);
  res.report.success = true;
  return res;
}

TrajectoryResult Solver::SolveTrajectory(
    const std::vector<Mat4>& targets,
    const Vec7& q_init,
    double psi_init) const {
  TrajectoryResult out;
  Vec7 q_prev = q_init;
  double psi_prev = psi_init;
  for (const auto& target : targets) {
    const IkResult res = IkArmAngle(target, q_prev, psi_prev);
    out.q_list.push_back(res.q_best);
    out.reports.push_back(res.report);
    if (!res.q_best.has_value()) {
      out.psi_list.push_back(psi_prev);
      continue;
    }
    q_prev = *res.q_best;
    psi_prev = res.report.psi_selected;
    out.psi_list.push_back(psi_prev);
  }
  return out;
}

}  // namespace super_ik_tsinghua_1dqp
