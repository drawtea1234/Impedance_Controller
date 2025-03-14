// Copyright (c) 2021 Franka Emika GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <franka_example_controllers/cartesian_impedance_controller.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>
#include "franka_example_controllers/pseudo_inversion.hpp"
#include <Eigen/Eigen>

namespace franka_example_controllers
{

controller_interface::InterfaceConfiguration CartesianImpedanceController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i)
  {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration CartesianImpedanceController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Creates state interface for robot state
  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names())
  {
    config.names.push_back(franka_robot_model_name);
  }
  // Creates state interface for robot model
  for (const auto& franka_robot_state_name : franka_robot_state_->get_state_interface_names())
  {
    config.names.push_back(franka_robot_state_name);
  }

  return config;
}

controller_interface::return_type CartesianImpedanceController::update(const rclcpp::Time& /*time*/,
                                                                       const rclcpp::Duration& /*period*/)
{
  // updateJointStates();
  // Vector7d q_goal = initial_q_;
  // auto time = this->get_node()->now() - start_time_;
  // double delta_angle = M_PI / 8.0 * (1 - std::cos(M_PI / 2.5 * time.seconds()));
  // q_goal(3) += delta_angle;
  // q_goal(4) += delta_angle;

  // const double kAlpha = 0.99;
  // dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;
  // Vector7d tau_d_calculated =
  //     k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);
  // for (int i = 0; i < num_joints; ++i) {
  //   command_interfaces_[i].set_value(tau_d_calculated(i));
  // }

  robot_state_ = franka_msgs::msg::FrankaRobotState();
  franka_robot_state_->get_values_as_message(robot_state_);

  //控制器不考虑惯性矩阵 std::array<double, 49> mass = franka_robot_model_->getMassMatrix();
  std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  // std::array<double, 42> jacobian_array =
  //   franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  auto jacobian_array = franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  pose_J7 = franka_robot_model_->getPoseMatrix(franka::Frame::kJoint7);
  // Eigen::Map<Eigen::Matrix<double, 7, 7>> M(mass.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());

  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);
  pseudoInverse(jacobian, jacobian_pinv);

  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state_.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state_.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(robot_state_.tau_j_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state_.o_t_ee.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());

  std::array<double, 49> mass = franka_robot_model_->getMassMatrix();
  M = Eigen::Map<Eigen::Matrix<double, 7, 7>>(mass.data());
  Lambda = (jacobian * M.inverse() * jacobian.transpose()).inverse();

  Eigen::Matrix<double, 6, 1> error, d_position_d;

  d_position_d.setZero();

  error.head(3) << position - position_d_;

  // orientation error
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0)
  {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation * orientation_d_.inverse());
  // convert to axis angle
  Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
  // compute "orientation error"
  error.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();

  std::ostringstream oss_d;
  oss_d << "error: [";
  for (int i = 0; i < error.size(); ++i)
  {
    oss_d << error(i);
    if (i < error.size() - 1)
    {
      oss_d << ", ";
    }
  }
  oss_d << "]";
  // RCLCPP_INFO(rclcpp::get_logger("Data"), "%s", oss_d.str().c_str());

  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7), null_vect(7), tau_joint_limit(7);

  tau_task << jacobian.transpose() *
                  ((1 - m_M) * (-cartesian_stiffness_ * error - cartesian_damping_ * ((jacobian * dq) - d_position_d)) -
                   m_M * O_F_ext_hat_K_M);  // double critic damping

  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) - jacobian.transpose() * jacobian_transpose_pinv) *
                       (nullspace_stiffness_ * (q_d_nullspace_ - q) - (2.0 * sqrt(nullspace_stiffness_)) * dq);

  tau_d << tau_task + coriolis + tau_nullspace;

  tau_d = saturateTorqueRate(tau_d, tau_J_d);

  ref_tau = -cartesian_stiffness_ * error - cartesian_damping_ * ((jacobian * dq) - d_position_d);

  for (int i = 0; i < tau_d.size(); ++i)
  {
    if (tau_d(i) > 3.0)
    {
      tau_d(i) = 3.0;
    }
    else if (tau_d(i) < -3.0)
    {
      tau_d(i) = -3.0;
    }
  }

  for (int i = 0; i < num_joints; i++)
  {
    command_interfaces_[i].set_value(tau_d[i]);
    std::cout << tau_d[i] << std::endl;
  }

  tau_d_ = tau_d;

  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target);
  position_d_ = filter_params_ * position_d_target + (1.0 - filter_params_) * position_d_;

  return controller_interface::return_type::OK;
}

CallbackReturn CartesianImpedanceController::on_init()
{
  try
  {
    auto_declare<std::string>("arm_id", "panda");
  }
  catch (const std::exception& e)
  {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }

  sub_equilibrium_pose_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      "equilibrium_pose", 10,
      std::bind(&CartesianImpedanceController::equilibriumPoseCallback, this, std::placeholders::_1));

  sub_equilibrium_force_ = get_node()->create_subscription<geometry_msgs::msg::WrenchStamped>(
      "your_namespace/force_torque_sensor_broadcaster/wrench", 10,
      std::bind(&CartesianImpedanceController::equilibriumForceCallback, this, std::placeholders::_1));

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/)
{
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  // translational_stiffness = get_node()->get_parameter("translational_stiffness").as_double();
  // rotational_stiffness = get_node()->get_parameter("rotational_stiffness").as_double();
  translational_stiffness = 150.0;
  rotational_stiffness = 10.0;

  T << 0.912818721, -0.19717224, -0.082037553;

  G << 0.03974507, -0.34600526, -5.93976337;
  F << -3.93321387, 8.53191934, -14.06126017;
  P << 0.005798487, 0.005741689, 0.023214488;

  cartesian_stiffness_.setZero();
  cartesian_stiffness_.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  cartesian_stiffness_.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  cartesian_damping_.setZero();
  cartesian_damping_.topLeftCorner(3, 3) << 2.0 * sqrt(translational_stiffness) * Eigen::MatrixXd::Identity(3, 3);
  cartesian_damping_.bottomRightCorner(3, 3) << 2.0 * sqrt(rotational_stiffness) * Eigen::MatrixXd::Identity(3, 3);

  franka_robot_state_ = std::make_unique<franka_semantic_components::FrankaRobotState>(
      franka_semantic_components::FrankaRobotState(arm_id_ + "/" + k_robot_state_interface_name));
  franka_robot_model_ =
      std::make_unique<franka_semantic_components::FrankaRobotModel>(franka_semantic_components::FrankaRobotModel(
          arm_id_ + "/" + k_robot_model_interface_name, arm_id_ + "/" + k_robot_state_interface_name));
  O_F_ext_hat_K_M.setZero();
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/)
{
  // updateJointStates();

  start_time_ = this->get_node()->now();

  franka_robot_state_->assign_loaned_state_interfaces(state_interfaces_);
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  init_robot_state_ = franka_msgs::msg::FrankaRobotState();
  franka_robot_state_->get_values_as_message(init_robot_state_);

  // get jacobian
  // std::array<double, 42> jacobian_array =
  //     franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);

  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(init_robot_state_.q.data());

  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(init_robot_state_.o_t_ee.data()));

  // set equilibrium point to current state
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  position_d_target = position_d_;
  orientation_d_target = orientation_d_;

  pose_d_orientation_quat_.coeffs() << orientation_d_.coeffs();
  pose_d_.head(3) << position_d_;
  pose_d_.tail(3) << pose_d_orientation_quat_.toRotationMatrix().eulerAngles(0, 1, 2);

  q_d_nullspace_ = q_initial;

  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
CartesianImpedanceController::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/)
{
  franka_robot_state_->release_interfaces();
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

void CartesianImpedanceController::updateJointStates()
{
  for (auto i = 0; i < num_joints; ++i)
  {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

Eigen::Matrix<double, 7, 1> CartesianImpedanceController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated, const Eigen::Matrix<double, 7, 1>& tau_J_d)
{  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++)
  {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void CartesianImpedanceController::equilibriumPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!msg)
  {
    RCLCPP_ERROR(get_node()->get_logger(), "Received null pointer in equilibriumPoseCallback");
    return;
  }

  // 检查四元数的有效性
  double q_norm =
      std::sqrt(msg->pose.orientation.x * msg->pose.orientation.x + msg->pose.orientation.y * msg->pose.orientation.y +
                msg->pose.orientation.z * msg->pose.orientation.z + msg->pose.orientation.w * msg->pose.orientation.w);
  if (std::abs(q_norm - 1.0) > 1e-6)
  {
    RCLCPP_WARN(get_node()->get_logger(), "Received invalid quaternion in equilibriumPoseCallback");
  }

  position_d_target << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;

  static Eigen::Quaterniond last_orientation_d_target(orientation_d_target);
  orientation_d_target.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z,
      msg->pose.orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target.coeffs()) < 0.0)
  {
    orientation_d_target.coeffs() *= -1;
  }
  last_orientation_d_target = orientation_d_target;
}

void CartesianImpedanceController::equilibriumForceCallback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
{
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(pose_J7.data()));

  Eigen::Quaterniond orientation(transform.rotation());

  Eigen::Matrix<double, 6, 1> O_F_ext_hat_K_M_;

  O_F_ext_hat_K_M_(0) = msg->wrench.force.x;
  O_F_ext_hat_K_M_(1) = msg->wrench.force.y;
  O_F_ext_hat_K_M_(2) = msg->wrench.force.z;
  O_F_ext_hat_K_M_(3) = msg->wrench.torque.x;
  O_F_ext_hat_K_M_(4) = msg->wrench.torque.y;
  O_F_ext_hat_K_M_(5) = msg->wrench.torque.z;

  // 获取旋转矩阵
  Eigen::Matrix3d R = orientation.toRotationMatrix();

  // 分离力和力矩
  Eigen::Matrix<double, 3, 1> force_sensor = O_F_ext_hat_K_M_.block<3, 1>(0, 0);
  Eigen::Matrix<double, 3, 1> torque_sensor = O_F_ext_hat_K_M_.block<3, 1>(3, 0);

  // 校正力和力矩
  Eigen::Matrix<double, 3, 1> force_world = R * (force_sensor + F);
  Eigen::Matrix<double, 3, 1> torque_world = R * (torque_sensor + T);
  Eigen::Matrix<double, 3, 1> force_corrected = force_world - G;
  Eigen::Matrix<double, 3, 1> P_World = R * P;
  // 计算重力导致的扭矩并校正
  Eigen::Vector3d gravity_torque_world = P_World.cross(G);
  Eigen::Matrix<double, 3, 1> torque_corrected = torque_world - gravity_torque_world;

  O_F_ext_hat_K_M << force_corrected, torque_corrected;

  // std::ostringstream oss_d;
  // oss_d << "error: [";
  // for (int i = 0; i < O_F_ext_hat_K_M.size(); ++i)
  // {
  //   oss_d << O_F_ext_hat_K_M(i);
  //   if (i < O_F_ext_hat_K_M.size() - 1)
  //   {
  //     oss_d << ", ";
  //   }
  // }
  // RCLCPP_INFO(rclcpp::get_logger("Data"), "%s", oss_d.str().c_str());

  // std::ostringstream oss__d;
  // oss__d << "error: [";
  // for (int i = 0; i < ref_tau.size(); ++i)
  // {
  //   oss__d << ref_tau(i);
  //   if (i < ref_tau.size() - 1)
  //   {
  //     oss__d << ", ";
  //   }
  // }
  // RCLCPP_INFO(rclcpp::get_logger("tau_task"), "%s", oss__d.str().c_str());

  // RCLCPP_INFO(rclcpp::get_logger("x"), "%f", orientation.x());
  // RCLCPP_INFO(rclcpp::get_logger("y"), "%f", orientation.y());
  // RCLCPP_INFO(rclcpp::get_logger("z"), "%f", orientation.z());
  // RCLCPP_INFO(rclcpp::get_logger("w"), "%f", orientation.w());
  //     // 打印或处理转换后的矩阵
  //     RCLCPP_INFO(get_node()->get_logger(),
  //                 "Calibrated forces and torques:\nfx: %f, fy: %f, fz: %f, tx: %f, ty: %f, tz: %f",
  //                 force_corrected(0), force_corrected(1), force_corrected(2),
  //                 torque_corrected(0), torque_corrected(1), torque_corrected(2));
  // } else {
  //     RCLCPP_WARN(get_node()->get_logger(), "Not enough data received.");
  // }

  //   // 打印或处理转换后的矩阵
  //   RCLCPP_INFO(get_node()->get_logger(),
  //               "Received forces and torques:\nfx: %f, fy: %f, fz: %f, tx: %f, ty: %f, tz: %f", O_F_ext_hat_K_M(0),
  //               O_F_ext_hat_K_M(1), O_F_ext_hat_K_M(2), O_F_ext_hat_K_M(3), O_F_ext_hat_K_M(4), O_F_ext_hat_K_M(5));
  // }
  // else
  // {
  //   RCLCPP_WARN(get_node()->get_logger(), "Not enough data received.");
  // }
}

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianImpedanceController,
                       controller_interface::ControllerInterface)