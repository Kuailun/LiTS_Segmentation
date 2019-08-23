import os
def CheckDirectory(p):
    """Check whether directory is existed, if not then create"""
    if os.path.exists(p):
        pass
    else:
        os.mkdir(p)
        pass
    pass