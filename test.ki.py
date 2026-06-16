import pandapipes as pp
import pandapipes.explain

import numpy as np

# pandapipes.explain.enable()


net = pp.create_empty_network(fluid="lgas")

j1 = pp.create_junction(net, 1.05, 293.15, height_m=0)
j2 = pp.create_junction(net, 1.05, 293.15, height_m=0)
j3 = pp.create_junction(net, 1.05, 293.15, height_m=0)
j4 = pp.create_junction(net, 1.05, 293.15, height_m=0)
j5 = pp.create_junction(net, 1.05, 293.15, height_m=0)
j6 = pp.create_junction(net, 1.05, 293.15, height_m=0)

pp.create_ext_grid(net, junction=j1, p_bar=1.1, t_k=293.15)

pp.create_pipe_from_parameters(net, j1, j2, length_km=10,k_mm=0.2, inner_diameter_mm=300)
pp.create_pipe_from_parameters(net, j2, j3, length_km=2, k_mm=0.2,  inner_diameter_mm=300)
pp.create_pipe_from_parameters(net, j2, j4, length_km=2.5,k_mm=0.2,  inner_diameter_mm=300)
pp.create_pipe_from_parameters(net, j3, j5, length_km=1, k_mm=0.2,inner_diameter_mm=300)
pp.create_pipe_from_parameters(net, j4, j6, length_km=1, k_mm=0.2, inner_diameter_mm=300)

pp.create_valve(net, junction=j5, element=j6, et="ju", inner_diameter_mm=50, opened=True)

pp.create_sink(net, junction=j4, mdot_kg_per_s=0.545)
pp.create_source(net, junction=j3, mdot_kg_per_s=0.234)
net.ext_grid.loc[0, "p_bar"] = 100000

pp.diagnostic(net)