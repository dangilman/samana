CODE_VERSION = '1.3.1'

def check_code_version(required_version):
    assert required_version == CODE_VERSION
def check_code_version_lenstronomy(required_version):
    import lenstronomy
    assert lenstronomy.__version__ == required_version
