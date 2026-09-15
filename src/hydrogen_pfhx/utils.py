import yaml
def tpd_to_kps(tonnes_per_day):
    return tonnes_per_day * 1000.0 / (24.0 * 3600.0)
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)
