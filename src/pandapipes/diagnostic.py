# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes as pp
import numpy as np

from pandapipes import PipeflowNotConverged
from pandapipes.idx_node import PINIT
from pandapipes.idx_branch import MDOTINIT

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def check_net(net, low_length_limit_km=0.01,
              check_scaling_factor=1e-5,
              diameter_increase_factor=2,
              heat_consumer_scaling_factor=0.1,
              deltat_scaling_factor=2):
    """

    Run diagnostic checks on the net to identify potential flaws.

    The checks apply selected modifications to copies of the input net and
    test whether pipeflow convergence improves. A PipeflowNotConverged
    exception is treated as a diagnostic result, while other exceptions
    indicate that the corresponding check could not be evaluated
    """

    net = net.deepcopy()  # do not modify the direct input
    try:
        pp.pipeflow(net)
        if net.converged:
            logger.info("The initial, unmodified pipeflow converges.")
        else:
            logger.warning("The initial, unmodified pipeflow does NOT converge.")
    except Exception as e:
        logger.info(f"The initial, unmodified pipeflow does NOT converge.\n"
                    f"\t\tThis exception is raised:\n\t\t{e}")

    # check ext_grid
    if net.fluid.is_gas & (not hasattr(net, "ext_grid") | net.ext_grid.empty):
        logger.warning("The net does not have an external grid! "
                       "An external grid is required for gas networks.")

    # check zero / low length
    zl = net.pipe.loc[net.pipe.length_km == 0]
    ll = net.pipe.loc[net.pipe.length_km <= low_length_limit_km]
    if not zl.empty:
        logger.warning(f"{len(zl.index)} pipes have a length of 0.0 km. (IDs: {zl.index})")
    if not ll.empty:
        logger.warning(f"{len(ll.index)} pipes have a length below"
                       f" {low_length_limit_km} km. "
                       f"This could lead to convergence issues. The lowest length in the net is "
                       f"{ll.length_km.min()} km.\n"
                       f"(IDs of pipelines with low length: {ll.index})")

        net2 = net.deepcopy()
        net2.pipe.loc[net2.pipe.length_km < low_length_limit_km].length_km = low_length_limit_km
        try:
            pp.pipeflow(net2)
            if net2.converged:
                logger.info(
                    f"If all short pipelines (< {low_length_limit_km} km) were set to "
                    f"{low_length_limit_km} km, the pipeflow would converge."
                )
        except PipeflowNotConverged:
            logger.info(
                f"Pipeflow still does not converge if all short pipelines "
                f"(< {low_length_limit_km} km) are set to {low_length_limit_km} km."
            )
        except Exception as e:
            logger.warning(
                "The short-pipeline-length check failed.\n"
                f"\t\tThe error message is: {e}"
            )

    # check iterations
    iterations = 200
    try:
        pp.pipeflow(net, iter=iterations)
        logger.info(f"The pipeflow converges after {net._internal_results['iterations']:d} "
                    f"iterations.")
    except PipeflowNotConverged:
        logger.info(f"After {iterations:d} iterations the pipeflow did NOT converge.")

    # check with little sink and source scaling
    if hasattr(net, "sink") or hasattr(net, "source"):
        logger.info("Testing with scaled-down sinks and sources.")

        net3 = net.deepcopy()

        if hasattr(net3, "sink"):
            net3.sink.scaling *= check_scaling_factor
        if hasattr(net3, "source"):
            net3.source.scaling *= check_scaling_factor
        try:
            pp.pipeflow(net3)
            if net3.converged:
                logger.info(
                    f"If sinks and sources were scaled by a factor of "
                    f"{check_scaling_factor}, the pipeflow would converge."
                )
        except PipeflowNotConverged:
            logger.info(
                f"Pipeflow still does not converge if sinks and sources are "
                f"scaled by a factor of {check_scaling_factor}."
            )
        except Exception as e:
            logger.warning(
                "The sink/source scaling check failed.\n"
                f"\t\tThe error message is: {e}"
            )

    # check k
    if any(net.pipe.k_mm > 0.5):
        logger.warning(f"Some pipes have a friction factor k_mm > 0.5 (extremely rough). The "
                       f"highest value in the net is {net.pipe.k_mm.max()}. Up to "
                       f"0.2 mm is a common value for old steel pipes."
                       f"\nRough pipes: {net.pipe.loc[net.pipe.k_mm > 0.5]}.")
    net4 = net.deepcopy()
    net4.pipe.k_mm = 1e-5

    try:
        pp.pipeflow(net4)
        if net4.converged:
            logger.info(
                "If the friction factor were reduced to 1e-5 for all pipes, "
                "the pipeflow would converge."
            )
    except PipeflowNotConverged:
        logger.info(
            "Pipeflow still does not converge if k_mm is reduced to 1e-5 "
            "for all pipes."
        )
    except Exception as e:
        logger.warning(
            "The pipe-roughness check failed.\n"
            f"\t\tThe error message is: {e}"
        )

    # check sink and source junctions:
    node_component = ["sink", "source", "ext_grid"]
    for nc in node_component:
        if hasattr(net, nc):
            missing = np.setdiff1d(net[nc].junction, net.junction.index)
            if len(missing):
                logger.warning(f"Some {nc}s are connected to non-existing junctions!"
                               f"\n{nc}s:{net[nc].loc[net[nc].junction.isin(missing)]}"
                               f"\nmissing junctions:{missing}")

    # check from and to junctions
    branch_component = ["pipe", "valve", "compressor", "pump", "heat_exchanger", "circulation_pump"]
    for bc in branch_component:
        if hasattr(net, bc):
            missing_f = np.setdiff1d(net[bc].from_junction, net.junction.index)
            missing_t = np.setdiff1d(net[bc].to_junction, net.junction.index)
            if len(missing_t) | len(missing_t):
                logger.warning(f"Some {bc}s are connected to non-existing junctions!")
                logger.warning(f"missing 'from' junctions:{missing_f}")
                logger.warning(f"missing 'to' junctions:{missing_t}")

    # check with increased pipe diameter
    net5 = net.deepcopy()

    if net5.fluid.is_gas:
        diameter_threshold_mm = 6
    else:
        diameter_threshold_mm = 20

    small_pipes = net5.pipe.inner_diameter_mm < diameter_threshold_mm
    if small_pipes.any():
        logger.info(
            f"Testing with pipe diameters below {diameter_threshold_mm} mm "
            f"increased by factor {diameter_increase_factor}."
        )
        net5.pipe.loc[small_pipes, "inner_diameter_mm"] *= diameter_increase_factor

        try:
            pp.pipeflow(net5)
            if net5.converged:
                logger.info(
                    f"If pipe diameters below {diameter_threshold_mm} mm were increased "
                    f"by factor {diameter_increase_factor}, the pipeflow would converge."
                )

        except PipeflowNotConverged:
            logger.info(
                f"Pipeflow still does not converge if pipe diameters below "
                f"{diameter_threshold_mm} mm are increased by factor "
                f"{diameter_increase_factor}."
            )

        except Exception as e:
            logger.warning(
                "The pipe-diameter check failed.\n"
                f"\t\tThe error message is: {e}"
            )

    # check heat transfer coefficient
    if any(net.pipe.u_w_per_m2k > 1):
        net6 = net.deepcopy()
        net6.pipe.loc[net6.pipe.u_w_per_m2k > 1, "u_w_per_m2k"] *= 0.1

        try:
            pp.pipeflow(net6)
            if net6.converged:
                logger.info(
                    "If pipe heat transfer coefficients above 1 W/(m²K) were reduced "
                    "by factor 0.1, the pipeflow would converge."
                )
        except PipeflowNotConverged:
            logger.info(
                "Pipeflow still does not converge if pipe heat transfer coefficients "
                "above 1 W/(m²K) are reduced by factor 0.1."
            )
        except Exception as e:
            logger.warning(
                "The heat-transfer-coefficient check failed.\n"
                f"\t\tThe error message is: {e}"
            )

    # check with all valves opened
    if not net.valve.empty and (~net.valve.opened).any():
        net7 = net.deepcopy()
        net7.valve.opened = True

        try:
            pp.pipeflow(net7)
            if net7.converged:
                logger.info(
                    "If all valves were opened, the pipeflow would converge."
                )
        except PipeflowNotConverged:
            logger.info(
                "Pipeflow still does not converge if all valves are opened."
            )
        except Exception as e:
            logger.warning(
                "The valve-opening check failed.\n"
                f"\t\tThe error message is: {e}"
            )

    # check heat consumer control parameters
    # The idea is to reduce the thermal load of the heat consumers.
    # Lower heat demand (qext_w) and mass flow (controlled_mdot_kg_per_s)
    # reduce the hydraulic and thermal stress on the network.
    # For configurations using deltat_k, the temperature difference is
    # increased to reduce the required mass flow according to
    # Q = m * cp * deltaT.
    if hasattr(net, "heat_consumer") and not net.heat_consumer.empty:
        logger.info("Testing with adjusted heat consumer control parameters.")
        net8 = net.deepcopy()

        # combination 1: qext_w + controlled_mdot_kg_per_s
        # reduce heat demand and mass flow
        mask_qext_mdot = (
                net8.heat_consumer.qext_w.notna()
                & net8.heat_consumer.controlled_mdot_kg_per_s.notna()
                & net8.heat_consumer.deltat_k.isna()
                & net8.heat_consumer.treturn_k.isna()
        )

        net8.heat_consumer.loc[mask_qext_mdot, "qext_w"] *= heat_consumer_scaling_factor
        net8.heat_consumer.loc[mask_qext_mdot, "controlled_mdot_kg_per_s"] *= heat_consumer_scaling_factor

        # combination 2: qext_w + deltat_k
        # reduce heat demand and increase deltaT
        mask_qext_deltat = (
                net8.heat_consumer.qext_w.notna()
                & net8.heat_consumer.controlled_mdot_kg_per_s.isna()
                & net8.heat_consumer.deltat_k.notna()
                & net8.heat_consumer.treturn_k.isna()
        )

        net8.heat_consumer.loc[mask_qext_deltat, "qext_w"] *= heat_consumer_scaling_factor

        net8.heat_consumer.loc[mask_qext_deltat, "deltat_k"] *= deltat_scaling_factor

        # combination 3: qext_w + treturn_k
        # reduce heat demand
        mask_qext_treturn = (
                net8.heat_consumer.qext_w.notna()
                & net8.heat_consumer.controlled_mdot_kg_per_s.isna()
                & net8.heat_consumer.deltat_k.isna()
                & net8.heat_consumer.treturn_k.notna()
        )

        net8.heat_consumer.loc[mask_qext_treturn, "qext_w"] *= heat_consumer_scaling_factor

        # combination 4: controlled_mdot_kg_per_s + deltat_k
        # reduce mass flow and increase deltaT
        mask_mdot_deltat = (
                net8.heat_consumer.qext_w.isna()
                & net8.heat_consumer.controlled_mdot_kg_per_s.notna()
                & net8.heat_consumer.deltat_k.notna()
                & net8.heat_consumer.treturn_k.isna()
        )

        net8.heat_consumer.loc[mask_mdot_deltat, "controlled_mdot_kg_per_s"] *= heat_consumer_scaling_factor
        net8.heat_consumer.loc[mask_mdot_deltat, "deltat_k"] *= deltat_scaling_factor

        # combination 5: controlled_mdot_kg_per_s + treturn_k
        # reduce mass flow
        mask_mdot_treturn = (
                net8.heat_consumer.qext_w.isna()
                & net8.heat_consumer.controlled_mdot_kg_per_s.notna()
                & net8.heat_consumer.deltat_k.isna()
                & net8.heat_consumer.treturn_k.notna()
        )

        net8.heat_consumer.loc[mask_mdot_treturn, "controlled_mdot_kg_per_s"] *= heat_consumer_scaling_factor
        try:
            pp.pipeflow(net8)
            if net8.converged:
                logger.info(
                    "If heat consumer control parameters were adjusted, "
                    "the pipeflow would converge."
                )
        except PipeflowNotConverged:
            logger.info(
                "Pipeflow still does not converge if heat consumer control "
                "parameters are adjusted."
            )
        except Exception as e:
            logger.warning(
                "The heat-consumer control-parameter check failed.\n"
                f"\t\tThe error message is: {e}"
            )

    # check with flattened junction heights
    if net.junction.height_m.nunique() > 1:
        net9 = net.deepcopy()
        net9.junction.height_m = 0

        try:
            pp.pipeflow(net9)
            if net9.converged:
                logger.info(
                    "If all junction heights were set to 0 m, "
                    "the pipeflow would converge."
                )
        except PipeflowNotConverged:
            logger.info(
                "Pipeflow still does not converge if all junction heights "
                "are set to 0 m."
            )
        except Exception as e:
            logger.warning(
                "The junction-height check failed.\n"
                f"\t\tThe error message is: {e}"
            )

    # check convergence in different calculation modes
    # Note: Heat mode cannot be executed independently because it requires
    # hydraulic results (node pressures and branch mass flows) as input.
    # Therefore, a hydraulic calculation is performed first and the resulting
    # PINIT and MDOTINIT values are passed to the heat calculation via sol_vec.
    for mode in ["hydraulics", "heat", "sequential", "bidirectional"]:
        net11 = net.deepcopy()

        try:
            if mode == "heat":
                pp.pipeflow(net11, mode="hydraulics")

                sol_vec = np.r_[
                    net11["_pit"]["node"][:, PINIT],
                    net11["_pit"]["branch"][:, MDOTINIT]
                ]
                pp.pipeflow(net11, mode="heat", sol_vec=sol_vec)
            else:
                pp.pipeflow(net11, mode=mode)

            if net11.converged:
                logger.info(f"The pipeflow converges in mode '{mode}'.")

        except PipeflowNotConverged:
            logger.info(f"Pipeflow still does not converge in mode '{mode}'.")

        except Exception as e:
            logger.warning(
                f"The calculation-mode check failed for mode '{mode}'.\n"
                f"\t\tThe error message is: {e}")

    # check with changed friction model
    for friction_model in ["nikuradse", "colebrook", "swamee-jain"]:
        net12 = net.deepcopy()

        try:
            pp.pipeflow(net12, friction_model=friction_model)
            if net12.converged:
                logger.info(
                    f"The pipeflow converges with friction model '{friction_model}'.")

        except PipeflowNotConverged:
            logger.info(
                f"Pipeflow still does not converge with friction model "
                f"'{friction_model}'.")
        except Exception as e:
            logger.warning(
                f"The friction-model check failed for '{friction_model}'.\n"
                f"\t\tThe error message is: {e}")

    # check convergence without compressor pressure lift
    if hasattr(net, "compressor") and not net.compressor.empty:

        net13 = net.deepcopy()
        net13.compressor.pressure_ratio = 1

        try:
            pp.pipeflow(net13)
            if net13.converged:
                logger.info(
                    "If compressor pressure ratios were set to 1, "
                    "the pipeflow would converge."
                )
        except PipeflowNotConverged:
            logger.info(
                "Pipeflow still does not converge if compressor pressure ratios "
                "are set to 1."
            )
        except Exception as e:
            logger.warning(
                "The compressor-pressure-ratio check failed.\n"
                f"\t\tThe error message is: {e}"
            )

    # check with inactive pressure controls
    if hasattr(net, "press_control") and not net.press_control.empty:
        net14 = net.deepcopy()
        net14.press_control.control_active = False

        try:
            pp.pipeflow(net14)
            if net14.converged:
                logger.info(
                    "If all pressure controls were deactivated, "
                    "the pipeflow would converge."
                )
        except PipeflowNotConverged:
            logger.info(
                "Pipeflow still does not converge if all pressure controls "
                "are deactivated."
            )
        except Exception as e:
            logger.warning(
                "The pressure-control check failed.\n"
                f"\t\tThe error message is: {e}"
            )


def pipeflow_alpha_sweep(net, **kwargs):
    """Run the pipeflow many times with different alpha (NR damping factor) settings between 0.1 and 1 in steps of 0.1"""
    net.converged = False
    alphas = [1]
    for i in range(1, 10):
        if i % 2 == 1:
            alphas.append(round(1 - (i // 2) * 0.1, 1))
        else:
            alphas.append(round((i // 2) * 0.1, 1))
    if kwargs is None:
        kwargs = {}

    for alpha in alphas:
        kwargs.update({"alpha": alpha})
        try:
            pp.pipeflow(net, **kwargs)
        except Exception as e:
            logger.debug(f"Pipeflow did not converge with alpha = {alpha}.\nError: {e}")
        if net.converged:
            logger.info(f"Pipeflow did converge with alpha = {alpha}.")
            return
    logger.warning(f"Pipeflow did not converge with any alpha in {alphas}.")