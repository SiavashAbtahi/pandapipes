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



class InitialPipeflowCheck:

    def diagnostic(self, net):
        net2 = net.deepcopy()

        try:
            pp.pipeflow(net2)
            return net2.converged

        except Exception as e:
            logger.error(f"Initial pipeflow check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.info(
                "The initial, unmodified pipeflow does NOT converge.\n"
                f"\t\tThis exception is raised:\n\t\t{error}"
            )
            return

        if result:
            logger.info("The initial, unmodified pipeflow converges.")
        else:
            logger.warning("The initial, unmodified pipeflow does NOT converge.")



# check ext_grid
class MissingExtGridCheck:

    def diagnostic(self, net):
        if net.fluid.is_gas and (
            not hasattr(net, "ext_grid") or net.ext_grid.empty
        ):
            return True

        return None

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Missing ext_grid check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result is None:
            return

        logger.warning(
            "The net does not have an external grid! "
            "An external grid is required for gas networks."
        )



# check zero / low length
class ShortPipeLengthCheck:

    def __init__(self, low_length_limit_km=0.01):
        self.low_length_limit_km = low_length_limit_km
        self.zero_length_pipes = None
        self.low_length_pipes = None

    def diagnostic(self, net):
        self.zero_length_pipes = net.pipe.loc[net.pipe.length_km == 0]
        self.low_length_pipes = net.pipe.loc[
            net.pipe.length_km <= self.low_length_limit_km
        ]

        if self.low_length_pipes.empty:
            return None

        net2 = net.deepcopy()
        net2.pipe.loc[
            net2.pipe.length_km < self.low_length_limit_km,
            "length_km"
        ] = self.low_length_limit_km

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception as e:
            logger.error(f"Short-pipeline-length check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Short-pipeline-length check failed due to the following error:"
            )
            logger.warning(error)
            return

        if self.zero_length_pipes is not None and not self.zero_length_pipes.empty:
            logger.warning(
                f"{len(self.zero_length_pipes.index)} pipes have a length of 0.0 km. "
                f"(IDs: {self.zero_length_pipes.index})"
            )

        if self.low_length_pipes is None or self.low_length_pipes.empty:
            logger.info("No pipes with zero or low length found.")
            return

        logger.warning(
            f"{len(self.low_length_pipes.index)} pipes have a length below "
            f"{self.low_length_limit_km} km. This could lead to convergence issues. "
            f"The lowest length in the net is {self.low_length_pipes.length_km.min()} km.\n"
            f"(IDs of pipelines with low length: {self.low_length_pipes.index})"
        )

        if result:
            logger.info(
                f"If all short pipelines (< {self.low_length_limit_km} km) were set to "
                f"{self.low_length_limit_km} km, the pipeflow would converge."
            )
        else:
            logger.info(
                f"Pipeflow still does not converge if all short pipelines "
                f"(< {self.low_length_limit_km} km) are set to {self.low_length_limit_km} km."
            )



# check iterations
class IterationCheck:

    def __init__(self, iterations=200):
        self.iterations = iterations
        self.required_iterations = None

    def diagnostic(self, net):
        try:
            net2 = net.deepcopy()
            pp.pipeflow(net2, iter=self.iterations)
            self.required_iterations = net2._internal_results["iterations"]

            return True

        except PipeflowNotConverged:
            return False

        except Exception as e:
            logger.error(f"Iteration check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Iteration check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result:
            logger.info(
                f"The pipeflow converges after "
                f"{self.required_iterations:d} iterations."
            )
        else:
            logger.info(
                f"After {self.iterations:d} iterations "
                f"the pipeflow did NOT converge."
            )


# check with little sink and source scaling
class SinkSourceScalingCheck:

    def __init__(self, scaling_factor=1e-5):
        self.scaling_factor = scaling_factor

    def diagnostic(self, net):
        if not hasattr(net, "sink") and not hasattr(net, "source"):
            return None

        net2 = net.deepcopy()

        if hasattr(net2, "sink"):
            net2.sink.scaling *= self.scaling_factor

        if hasattr(net2, "source"):
            net2.source.scaling *= self.scaling_factor

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception as e:
            logger.error(f"Sink/source scaling check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Sink/source scaling check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result is None:
            logger.info(
                "No sinks or sources found in the network."
            )
            return

        if result:
            logger.info(
                f"If sinks and sources were scaled by a factor of "
                f"{self.scaling_factor}, the pipeflow would converge."
            )
        else:
            logger.info(
                f"Pipeflow still does not converge if sinks and sources are "
                f"scaled by a factor of {self.scaling_factor}."
            )

# check k
class PipeRoughnessCheck:

    def __init__(self, reduced_k_mm=1e-5, roughness_limit_mm=0.5):
        self.reduced_k_mm = reduced_k_mm
        self.roughness_limit_mm = roughness_limit_mm
        self.rough_pipes = None

    def diagnostic(self, net):
        self.rough_pipes = net.pipe.loc[net.pipe.k_mm > self.roughness_limit_mm]

        net2 = net.deepcopy()
        net2.pipe.k_mm = self.reduced_k_mm

        try:
            pp.pipeflow(net2)
            return net2.converged
        except PipeflowNotConverged:
            return False
        except Exception as e:
            logger.error(f"Pipe-roughness check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.warning("Pipe-roughness check failed due to the following error:")
            logger.warning(error)
            return

        if self.rough_pipes is not None and not self.rough_pipes.empty:
            logger.warning(
                f"Some pipes have a friction factor k_mm > {self.roughness_limit_mm} "
                f"(extremely rough). The highest value in the net is "
                f"{self.rough_pipes.k_mm.max()}. Up to 0.2 mm is a common value "
                f"for old steel pipes.\nRough pipes: {self.rough_pipes}."
            )

        if result:
            logger.info(
                f"If the friction factor were reduced to {self.reduced_k_mm} for all pipes, "
                "the pipeflow would converge."
            )
        else:
            logger.info(
                f"Pipeflow still does not converge if k_mm is reduced to {self.reduced_k_mm} "
                "for all pipes."
            )

# check sink and source junctions:
class MissingNodeJunctionsCheck:

    def diagnostic(self, net):
        node_components = ["sink", "source", "ext_grid"]
        result = {}

        for nc in node_components:
            if hasattr(net, nc):
                missing = np.setdiff1d(net[nc].junction, net.junction.index)

                if len(missing):
                    result[nc] = {
                        "missing_junctions": missing,
                        "elements": net[nc].loc[net[nc].junction.isin(missing)]
                    }

        return result if result else None

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Node-junction check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result is None:
            logger.info(
                "No node components connected to non-existing junctions found."
            )
            return

        for nc, values in result.items():
            logger.warning(
                f"Some {nc}s are connected to non-existing junctions!"
                f"\n{nc}s:{values['elements']}"
                f"\nmissing junctions:{values['missing_junctions']}"
            )


# check from and to junctions
class MissingBranchJunctionsCheck:

    def diagnostic(self, net):
        branch_components = ["pipe", "valve", "compressor", "pump", "heat_exchanger", "circulation_pump"]
        result = {}

        for bc in branch_components:
            if hasattr(net, bc):
                missing_f = np.setdiff1d(net[bc].from_junction, net.junction.index)
                missing_t = np.setdiff1d(net[bc].to_junction, net.junction.index)

                if len(missing_f) or len(missing_t):
                    result[bc] = {
                        "missing_from_junctions": missing_f,
                        "missing_to_junctions": missing_t
                    }

        return result if result else None

    def report(self, error, result):
        if error is not None:
            logger.warning("Branch-junction check failed due to the following error:")
            logger.warning(error)
            return

        if result is None:
            logger.info("No branch components connected to non-existing junctions found.")
            return

        for bc, values in result.items():
            logger.warning(f"Some {bc}s are connected to non-existing junctions!")
            logger.warning(f"missing 'from' junctions: {values['missing_from_junctions']}")
            logger.warning(f"missing 'to' junctions: {values['missing_to_junctions']}")


# check with increased pipe diameter
class PipeDiameterCheck:

    def __init__(self, diameter_increase_factor=2):
        self.diameter_increase_factor = diameter_increase_factor
        self.diameter_threshold_mm = None

    def diagnostic(self, net):
        net2 = net.deepcopy()

        if net2.fluid.is_gas:
            self.diameter_threshold_mm = 6
        else:
            self.diameter_threshold_mm = 20

        small_pipes = net2.pipe.inner_diameter_mm < self.diameter_threshold_mm

        if not small_pipes.any():
            return None

        net2.pipe.loc[small_pipes, "inner_diameter_mm"] *= self.diameter_increase_factor

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception as e:
            logger.error(f"Pipe-diameter check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.warning("Pipe-diameter check failed due to the following error:")
            logger.warning(error)
            return

        if result is None:
            logger.info("No pipe diameters below the threshold found.")
            return

        if result:
            logger.info(
                f"If pipe diameters below {self.diameter_threshold_mm} mm were increased "
                f"by factor {self.diameter_increase_factor}, the pipeflow would converge."
            )
        else:
            logger.info(
                f"Pipeflow still does not converge if pipe diameters below "
                f"{self.diameter_threshold_mm} mm are increased by factor "
                f"{self.diameter_increase_factor}."
            )

# check heat transfer coefficient
class HeatTransferCoefficientCheck:

    def diagnostic(self, net):
        if not any(net.pipe.u_w_per_m2k > 1):
            return None

        net2 = net.deepcopy()
        net2.pipe.loc[
            net2.pipe.u_w_per_m2k > 1,
            "u_w_per_m2k"
        ] *= 0.1

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception as e:
            logger.error(f"Heat-transfer-coefficient check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Heat-transfer-coefficient check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result is None:
            logger.info(
                "No pipe heat transfer coefficients above 1 W/(m²K) found."
            )
            return

        if result:
            logger.info(
                "If pipe heat transfer coefficients above 1 W/(m²K) were reduced "
                "by factor 0.1, the pipeflow would converge."
            )
        else:
            logger.info(
                "Pipeflow still does not converge if pipe heat transfer coefficients "
                "above 1 W/(m²K) are reduced by factor 0.1."
            )

# check with all valves opened
class ValveOpeningCheck:

    def diagnostic(self, net):
        if net.valve.empty or not (~net.valve.opened).any():
            return None

        net2 = net.deepcopy()
        net2.valve.opened = True

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception as e:
            logger.error(f"Valve-opening check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Valve-opening check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result is None:
            logger.info(
                "No closed valves found in the network."
            )
            return

        if result:
            logger.info(
                "If all valves were opened, the pipeflow would converge."
            )
        else:
            logger.info(
                "Pipeflow still does not converge if all valves are opened."
            )

# check heat consumer control parameters
# The idea is to reduce the thermal load of the heat consumers.
# Lower heat demand (qext_w) and mass flow (controlled_mdot_kg_per_s)
# reduce the hydraulic and thermal stress on the network.
# For configurations using deltat_k, the temperature difference is
# increased to reduce the required mass flow according to
# Q = m * cp * deltaT.
class HeatConsumerControlParameterCheck:

    def __init__(self, heat_consumer_scaling_factor=0.1, deltat_scaling_factor=2):
        self.heat_consumer_scaling_factor = heat_consumer_scaling_factor
        self.deltat_scaling_factor = deltat_scaling_factor
        self.affected_heat_consumers = None

    def diagnostic(self, net):
        if not hasattr(net, "heat_consumer") or net.heat_consumer.empty:
            return None

        net2 = net.deepcopy()
        hc = net2.heat_consumer

        # combination 1: qext_w + controlled_mdot_kg_per_s
        # reduce heat demand and mass flow
        mask_qext_mdot = (
            hc.qext_w.notna()
            & hc.controlled_mdot_kg_per_s.notna()
            & hc.deltat_k.isna()
            & hc.treturn_k.isna()
        )

        # combination 2: qext_w + deltat_k
        # reduce heat demand and increase deltaT
        mask_qext_deltat = (
            hc.qext_w.notna()
            & hc.controlled_mdot_kg_per_s.isna()
            & hc.deltat_k.notna()
            & hc.treturn_k.isna()
        )

        # combination 3: qext_w + treturn_k
        # reduce heat demand
        mask_qext_treturn = (
            hc.qext_w.notna()
            & hc.controlled_mdot_kg_per_s.isna()
            & hc.deltat_k.isna()
            & hc.treturn_k.notna()
        )

        # combination 4: controlled_mdot_kg_per_s + deltat_k
        # reduce mass flow and increase deltaT
        mask_mdot_deltat = (
            hc.qext_w.isna()
            & hc.controlled_mdot_kg_per_s.notna()
            & hc.deltat_k.notna()
            & hc.treturn_k.isna()
        )

        # combination 5: controlled_mdot_kg_per_s + treturn_k
        # reduce mass flow
        mask_mdot_treturn = (
            hc.qext_w.isna()
            & hc.controlled_mdot_kg_per_s.notna()
            & hc.deltat_k.isna()
            & hc.treturn_k.notna()
        )

        affected_mask = (
            mask_qext_mdot
            | mask_qext_deltat
            | mask_qext_treturn
            | mask_mdot_deltat
            | mask_mdot_treturn
        )

        self.affected_heat_consumers = hc.index[affected_mask].tolist()

        hc.loc[mask_qext_mdot, "qext_w"] *= self.heat_consumer_scaling_factor
        hc.loc[mask_qext_mdot, "controlled_mdot_kg_per_s"] *= self.heat_consumer_scaling_factor

        hc.loc[mask_qext_deltat, "qext_w"] *= self.heat_consumer_scaling_factor
        hc.loc[mask_qext_deltat, "deltat_k"] *= self.deltat_scaling_factor

        hc.loc[mask_qext_treturn, "qext_w"] *= self.heat_consumer_scaling_factor

        hc.loc[mask_mdot_deltat, "controlled_mdot_kg_per_s"] *= self.heat_consumer_scaling_factor
        hc.loc[mask_mdot_deltat, "deltat_k"] *= self.deltat_scaling_factor

        hc.loc[mask_mdot_treturn, "controlled_mdot_kg_per_s"] *= self.heat_consumer_scaling_factor

        try:
            pp.pipeflow(net2)
            return net2.converged
        except PipeflowNotConverged:
            return False
        except Exception as e:
            logger.error(f"Heat-consumer control-parameter check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Heat-consumer control-parameter check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result is None:
            logger.info("No heat consumers found in the network.")
            return

        logger.info("Testing with adjusted heat consumer control parameters.")

        if self.affected_heat_consumers is not None:
            logger.info(f"Adjusted heat consumer IDs: {self.affected_heat_consumers}")

        if result:
            logger.info(
                "If heat consumer control parameters were adjusted, "
                "the pipeflow would converge."
            )
        else:
            logger.info(
                "Pipeflow still does not converge if heat consumer control "
                "parameters are adjusted."
            )

# check with flattened junction heights
class JunctionHeightCheck:

    def diagnostic(self, net):
        if net.junction.height_m.nunique() <= 1:
            return None

        net2 = net.deepcopy()
        net2.junction.height_m = 0

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception as e:
            logger.error(f"Junction-height check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Junction-height check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result is None:
            logger.info(
                "All junction heights are already identical."
            )
            return

        if result:
            logger.info(
                "If all junction heights were set to 0 m, "
                "the pipeflow would converge."
            )
        else:
            logger.info(
                "Pipeflow still does not converge if all junction heights "
                "are set to 0 m."
            )


# check convergence in different calculation modes
# Note: Heat mode cannot be executed independently because it requires
# hydraulic results (node pressures and branch mass flows) as input.
# Therefore, a hydraulic calculation is performed first and the resulting
# PINIT and MDOTINIT values are passed to the heat calculation via sol_vec.
class CalculationModeCheck:

    def __init__(self, modes=None):
        self.modes = modes or ["hydraulics", "heat", "sequential", "bidirectional"]

    def diagnostic(self, net):
        results = {}

        for mode in self.modes:
            net2 = net.deepcopy()

            try:
                if mode == "heat":
                    pp.pipeflow(net2, mode="hydraulics")
                    sol_vec = np.r_[
                        net2["_pit"]["node"][:, PINIT],
                        net2["_pit"]["branch"][:, MDOTINIT]
                    ]
                    pp.pipeflow(net2, mode="heat", sol_vec=sol_vec)
                else:
                    pp.pipeflow(net2, mode=mode)
                results[mode] = net2.converged

            except PipeflowNotConverged:
                results[mode] = False

            except Exception as e:
                logger.error(
                    f"Calculation-mode check failed for mode '{mode}': {e}"
                )
                raise e

        return results if results else None

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Calculation-mode check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result is None:
            logger.info("No calculation modes were tested.")
            return

        for mode, converged in result.items():
            if converged:
                logger.info(f"The pipeflow converges in mode '{mode}'.")
            else:
                logger.info(f"Pipeflow still does not converge in mode '{mode}'.")



# check with changed friction model
class FrictionModelCheck:

    def __init__(self, friction_models=None):
        self.friction_models = friction_models or ["nikuradse", "colebrook", "swamee-jain"]

    def diagnostic(self, net):
        results = {}

        for friction_model in self.friction_models:
            net2 = net.deepcopy()

            try:
                pp.pipeflow(net2, friction_model=friction_model)
                results[friction_model] = net2.converged

            except PipeflowNotConverged:
                results[friction_model] = False

            except Exception as e:
                logger.error(
                    f"Friction-model check failed for '{friction_model}': {e}"
                )
                raise e

        return results if results else None

    def report(self, error, result):
        if error is not None:
            logger.warning("Friction-model check failed due to the following error:")
            logger.warning(error)
            return

        if result is None:
            logger.info("No friction models were tested.")
            return

        for friction_model, converged in result.items():
            if converged:
                logger.info(
                    f"The pipeflow converges with friction model '{friction_model}'."
                )
            else:
                logger.info(
                    f"Pipeflow still does not converge with friction model "
                    f"'{friction_model}'."
                )


# check convergence without compressor pressure lift
class CompressorPressureRatioCheck:

    def diagnostic(self, net):
        if not hasattr(net, "compressor") or net.compressor.empty:
            return None

        net2 = net.deepcopy()
        net2.compressor.pressure_ratio = 1

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception as e:
            logger.error(f"Compressor pressure-ratio check failed: {e}")
            raise e

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Compressor pressure-ratio check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result is None:
            logger.info("No compressors found in the network.")
            return

        if result:
            logger.info(
                "If compressor pressure ratios were set to 1, "
                "the pipeflow would converge."
            )
        else:
            logger.info(
                "Pipeflow still does not converge if compressor pressure ratios "
                "are set to 1."
            )



# check with inactive pressure controls
class InactivePressureControlsCheck:

    def diagnostic(self, net):
        if not hasattr(net, "press_control") or net.press_control.empty:
            return None

        net2 = net.deepcopy()
        net2.press_control.control_active = False

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception as e:
            logger.warning(
                "The pressure-control check failed.\n"
                f"\t\tThe error message is: {e}"
            )
            return None

    def report(self, error, result):
        if error is not None:
            logger.warning(
                "Inactive-pressure-controls check failed due to the following error:"
            )
            logger.warning(error)
            return

        if result is None:
            logger.info("No pressure controls found in the network.")
            return

        if result:
            logger.info(
                "If all pressure controls were deactivated, "
                "the pipeflow would converge."
            )
        else:
            logger.info(
                "Pipeflow still does not converge if all pressure controls "
                "are deactivated."
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