import numpy as np
dll=np.ctypeslib.load_library('hello','.')
dll.main()