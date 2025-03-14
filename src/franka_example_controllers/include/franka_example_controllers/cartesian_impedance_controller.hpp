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

#pragma once

#include <string>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "franka_semantic_components/franka_robot_model.hpp"
#include "franka_semantic_components/franka_robot_state.hpp"
#include <franka_msgs/msg/franka_robot_state.hpp>
#define IDENTITY Eigen::MatrixXd::Identity(6, 6)

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers
{

/**
 * The joint impedance example controller moves joint 4 and 5 in a very compliant periodic movement.
 */
class CartesianImpedanceController : public controller_interface::ControllerInterface
{
public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

private:
  std::string arm_id_;
  const std::string k_robot_state_interface_name{ "robot_state" };
  const std::string k_robot_model_interface_name{ "robot_model" };
  const int num_joints = 7;
  Vector7d q_;
  Vector7d initial_q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  //   Vector7d k_gains_;
  //   Vector7d d_gains_;
  rclcpp::Time start_time_;
  void updateJointStates();

  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;

  double translational_stiffness;
  double rotational_stiffness;
  double nullspace_stiffness_{ 20.0 };

  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  Eigen::Vector3d position_d_target;
  Eigen::Quaterniond orientation_d_target;

  const double delta_tau_max_{ 0.2 };

  Eigen::Matrix<double, 7, 1> saturateTorqueRate(const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
                                                 const Eigen::Matrix<double, 7, 1>& tau_j_d);

  Eigen::VectorXd tau_d_ = Eigen::VectorXd::Zero(7);
  Eigen::Matrix<double, 6, 1> ref_tau;
  std::array<double, 16> pose_J7;
  Eigen::Matrix<double, 6, 1> O_F_ext_hat_K_M;
  double filter_params_{ 0.01 };
  Eigen::Matrix<double, 6, 1> pose_d_;
  Eigen::Quaterniond pose_d_orientation_quat_;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  std::unique_ptr<franka_semantic_components::FrankaRobotState> franka_robot_state_;
  franka_msgs::msg::FrankaRobotState robot_state_, init_robot_state_;

  double m_M { 0.03};

  Eigen::Matrix<double, 3, 1> G;

  Eigen::Matrix<double, 3, 1> F;

  Eigen::Matrix<double, 3, 1> P;

  Eigen::Matrix<double, 3, 1> T;

  Eigen::MatrixXd jacobian_pinv;
  Eigen::MatrixXd jacobian_transpose_pinv;

  Eigen::Matrix<double, 7, 7> M;
  Eigen::Matrix<double, 6, 6> Lambda = IDENTITY;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_equilibrium_pose_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr sub_equilibrium_force_;
  void equilibriumPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void equilibriumForceCallback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg);
};

}  // namespace franka_example_controllers