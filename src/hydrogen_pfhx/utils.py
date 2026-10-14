"""Small helpers: reading configuration files and unit conversion."""

import yaml

def load_config(configuration_file_path):
    """Read a YAML configuration file into a dict (used by test_model.py and
    any script that runs the model).
    """
    with open(configuration_file_path, "r") as stream:
        try:
            configuration = yaml.safe_load(stream)
        # report the file name with any YAML syntax error
        except yaml.YAMLError as exc:
            raise ValueError('cannot parse %s: %s' % (configuration_file_path, exc)) from exc

    return configuration

def tpd_to_kps(mass_flow_rate):
    """Convert a mass flow from tonnes per day (configuration unit) to kg/s."""
    return mass_flow_rate * 1e3 / (60**2*24)
