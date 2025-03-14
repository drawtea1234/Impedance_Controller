// Copyright (c) 2022, Grzegorz Bartyzel
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the {copyright_holder} nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "net_ft_diagnostic_broadcaster/net_ft_diagnostic_broadcaster.hpp"

#include "rclcpp/rclcpp.hpp"
#include "diagnostic_updater/diagnostic_status_wrapper.hpp"

namespace net_ft_diagnostic_broadcaster
{
NetFTDiagnosticBroadcaster::NetFTDiagnosticBroadcaster()
{
}

controller_interface::CallbackReturn NetFTDiagnosticBroadcaster::on_init()
{
  try {
    auto_declare<double>("diagnostic_publish_rate", 1.0);
  } catch (std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  last_packet_count_ = 0;
  diagnostic_publisher_.reset();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration NetFTDiagnosticBroadcaster::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration command_interfaces_config;
  command_interfaces_config.type = controller_interface::interface_configuration_type::NONE;
  return command_interfaces_config;
}

controller_interface::InterfaceConfiguration NetFTDiagnosticBroadcaster::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  config.names.emplace_back("diagnostic/packet_count");
  config.names.emplace_back("diagnostic/lost_packets");
  config.names.emplace_back("diagnostic/out_of_order_count");
  config.names.emplace_back("diagnostic/status");
  return config;
}

controller_interface::CallbackReturn
NetFTDiagnosticBroadcaster::on_configure(const rclcpp_lifecycle::State& /*previous_state*/)
{
  if (!get_node()->get_parameter("diagnostic_publish_rate", publish_rate_)) {
    RCLCPP_INFO(get_node()->get_logger(), "Parameter 'diagnostic_publish_rate' not set");
    return controller_interface::CallbackReturn::ERROR;
  } else {
    RCLCPP_INFO(get_node()->get_logger(), "Publisher rate set to : %.1f Hz", publish_rate_);
  }
  diagnostic_publisher_ = get_node()->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "~/net_ft_diagnostic", rclcpp::SystemDefaultsQoS());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
NetFTDiagnosticBroadcaster::on_activate(const rclcpp_lifecycle::State& /*previous_state*/)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
NetFTDiagnosticBroadcaster::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type NetFTDiagnosticBroadcaster::update(const rclcpp::Time& /*time*/,
                                                                     const rclcpp::Duration& /*period*/)
{
  publish_diagnostic();
  return controller_interface::return_type::OK;
}

void NetFTDiagnosticBroadcaster::publish_diagnostic()
{
  diag_array_.status.clear();
  diagnostic_updater::DiagnosticStatusWrapper diag_status;

  auto packet_count = static_cast<uint32_t>(state_interfaces_[0].get_value());
  auto lost_packets = static_cast<uint32_t>(state_interfaces_[1].get_value());
  auto status = static_cast<uint32_t>(state_interfaces_[2].get_value());
  auto out_of_order_count = static_cast<uint32_t>(state_interfaces_[3].get_value());

  if (last_packet_count_ == packet_count) {
    diag_status.mergeSummary(diagnostic_updater::DiagnosticStatusWrapper::ERROR, "No new data received!");
  }
  if (status != 0) {
    diag_status.mergeSummaryf(diagnostic_updater::DiagnosticStatusWrapper::ERROR, "Net F/T driver reports error 0x%08x",
                              status);
  }
  diag_status.clear();
  diag_status.addf("System status", "0x%08x", status);
  diag_status.addf("Good packets", "%u", packet_count);
  diag_status.addf("Lost packets", "%u", lost_packets);
  diag_status.addf("Out-of-order packets", "%u", out_of_order_count);

  last_packet_count_ = packet_count;
  diag_array_.status.push_back(diag_status);
  diag_array_.header.stamp = get_node()->get_clock()->now();

  diagnostic_publisher_->publish(diag_array_);
}
}  // namespace net_ft_diagnostic_broadcaster

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(net_ft_diagnostic_broadcaster::NetFTDiagnosticBroadcaster,
                       controller_interface::ControllerInterface)
