"""Analytic Jacobian df/dy of bvp_model.bvp_function, passed to solve_bvp as
fun_jac (Jacobian variants) instead of its finite-difference estimate.

At every node the streams are converted to dual numbers (dualize), and the
same model code as the right-hand side (hexs, kinetics, energy and species
balances) is evaluated on them; the gradients form the node's Jacobian
block. Because the right-hand side couples streams only within a node,
df/dy is block-diagonal over the mesh.
"""


import numpy as np
from hydrogen_pfhx import hexs, heat_transfer_models, ortho_para_dynamics
from hydrogen_pfhx.dual import Dual, install_shim
from hydrogen_pfhx.dualize import dualize_stream
from hydrogen_pfhx import eos_derivatives as ed

# make the model modules dual-number aware (once, at import)
install_shim(hexs, heat_transfer_models, ortho_para_dynamics)


def _heat_of_conversion_dual(stream, topology, label):
    """Heat of conversion of a hydrogen stream as a dual number: its analytic
    T and P derivatives from the ortho and para equations of state.
    """
    T = stream.temperature
    rho_o, rho_p = stream.ortho_density, stream.para_density
    ortho, para = stream.ortho, stream.para

    # value, and the chain rule through each form's density at constant P
    hoc_val = para.enthalpy(rho_p, T) - ortho.enthalpy(rho_o, T)

    drho_o_dT = ed.drho_dT_P(ortho, rho_o, T); drho_o_dP = ed.drho_dP_T(ortho, rho_o, T)
    drho_p_dT = ed.drho_dT_P(para, rho_p, T);   drho_p_dP = ed.drho_dP_T(para, rho_p, T)

    dho_dT = ed.denthalpy_dT_rho(ortho, rho_o, T) + ed.denthalpy_drho_T(ortho, rho_o, T) * drho_o_dT
    dho_dP = ed.denthalpy_drho_T(ortho, rho_o, T) * drho_o_dP
    dhp_dT = ed.denthalpy_dT_rho(para, rho_p, T) + ed.denthalpy_drho_T(para, rho_p, T) * drho_p_dT
    dhp_dP = ed.denthalpy_drho_T(para, rho_p, T) * drho_p_dP

    e = topology.index[label]
    der = np.zeros(topology.n_states)
    der[e['T']] = dhp_dT - dho_dT
    # pressure state is in kPa
    der[e['P']] = (dhp_dP - dho_dP) * 1000.0
    return Dual(hoc_val, der)


def _node_jacobian(reactor, catalyst, streams, topology):
    """Jacobian block (n_states x n_states) of the right-hand side at one node,
    by evaluating the balances of bvp_function on dual-number streams.
    """
    n = topology.n_states

    duals = {}
    for label in topology.labels:
        e = topology.index[label]
        # dual-number copy of every stream at this node
        duals[label] = dualize_stream(
            streams[label], n, e['xp'], e['P'], e['T'])

    for label in topology.labels:
        d = duals[label]
        if topology.is_packed(label):
            d.calculate_velocity(
                reactor.total_side_area(label) * catalyst.void_fraction)
        else:
            d.calculate_velocity(reactor.total_side_area(label))
            d.calculate_reynolds_number(reactor.offset_strip_hydraulic_diameter)

    # transport and interface duties on the dual streams, as in bvp_function
    temperatures = {l: duals[l].temperature for l in topology.labels}
    for label in topology.labels:
        reactor.transport_behaviour(label, duals[label], catalyst, temperatures)

    duties, _ = reactor.calculate_heat_transfer_duty(temperatures)

    # each balance's gradient is one row of the block
    J = np.zeros((n, n))
    for label in topology.labels:
        e = topology.index[label]
        d = duals[label]

        heat_gained = 0.0
        for (a, b), q in duties.items():
            if a == label:
                heat_gained = heat_gained - q
            elif b == label:
                heat_gained = heat_gained + q

        if topology.is_packed(label):
            # species balance and conversion heat (rate per packed-duct volume)
            r_dot = ortho_para_dynamics.first_order_kinetics(d)
            dNpdz = r_dot * reactor.total_side_area(label)
            dxpdz = topology.flow_sign(label) * dNpdz * d.molecular_mass / d.mass_flow_rate
            J[e['xp'], :] = dxpdz.der
            hoc = _heat_of_conversion_dual(streams[label], topology, label)
            heat_gained = heat_gained - hoc * dNpdz

        # energy and momentum balances
        dTdz = topology.flow_sign(label) * heat_gained / (
            d.mass_flow_rate * d.specific_heat_capacity)
        J[e['T'], :] = dTdz.der

        dPdz = reactor.pressure_drop(label, d, catalyst) / 1e3
        J[e['P'], :] = dPdz.der

    return J


def fun_jac(x, y, additional_parameters):
    """df/dy for solve_bvp: array (n, n, m) over the m mesh nodes, in solver
    (physical) units.
    """
    reactor, catalyst, streams, topology, boundary_properties = additional_parameters
    # set the (float) stream states of every node, then differentiate around them
    from hydrogen_pfhx.bvp_model_physical import update_parameters

    n = topology.n_states
    m = y.shape[1]
    df_dy = np.zeros((n, n, m))
    for zi in range(m):
        streams = update_parameters(y[:, zi], streams, topology, boundary_properties)
        df_dy[:, :, zi] = _node_jacobian(reactor, catalyst, streams, topology)
    return df_dy


def normalized_fun_jac(x, y, additional_parameters, get_state_scales):
    """df/dy for the normalised variants: evaluated in physical units, then
    rescaled to the dimensionless states (J_ij * scale_j / scale_i).
    """
    reactor, catalyst, streams, topology, boundary_properties = additional_parameters
    scales = get_state_scales(topology, boundary_properties)
    y_physical = y * scales[:, np.newaxis]

    J_physical = fun_jac(x, y_physical, additional_parameters)

    # d(y_i/s_i)/d(y_j/s_j) = J_ij s_j / s_i
    rescale = scales[np.newaxis, :] / scales[:, np.newaxis]
    return J_physical * rescale[:, :, np.newaxis]
