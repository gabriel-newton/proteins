## Pipeline Constants
## Hard-coded configuration values for the PROT pipeline.

TORSION_INVARIANTS = ['tau_NA', 'tau_AC', 'tau_CN']
NON_TORSION_INVARIANTS = ['length_N', 'length_A', 'length_C', 'angle_N', 'angle_A', 'angle_C']
ALL_INVARIANTS = TORSION_INVARIANTS + NON_TORSION_INVARIANTS

INVARIANT_SHORTHAND = {
    'tau_NA': '$\phi$',
    'tau_AC': '$\psi$',
    'tau_CN': '$\omega$',
    'angle_N': '$\\alpha(N)$',
    'angle_A': '$\\alpha(A)$',
    'angle_C': '$\\alpha(C)$',
    'length_N': '$L(N)$',
    'length_A': '$L(A)$',
    'length_C': '$L(C)$'
}

INVARIANT_TYPES = {}
for inv in TORSION_INVARIANTS:
    INVARIANT_TYPES[inv] = 'torsion'
for inv in ['angle_N', 'angle_A', 'angle_C']:
    INVARIANT_TYPES[inv] = 'angle'
for inv in ['length_N', 'length_A', 'length_C']:
    INVARIANT_TYPES[inv] = 'length'

RESOLUTION_LEVELS = ['level_1'] # Chose "level" to stop resolution and residue confusion

RESOLUTION_BINS = {
    'level_1': {
        'torsion': 1.0,  # degrees
        'angle':   1.0,  # degrees
        'length':  0.05  # angstroms
    }
}

RESIDUE_CONTEXTS = [
    "Any", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
] # Amino Acid Labels, with an Any option

INVARIANT_LIMITS = {
    'tau_NA': {'limit_min': -180.0, 'limit_max': 180.0},
    'tau_AC': {'limit_min': -180.0, 'limit_max': 180.0},
    'tau_CN': {'limit_min': -180.0, 'limit_max': 180.0},
    'angle_N': {'limit_min': -180.0, 'limit_max': 180.0},
    'angle_A': {'limit_min': -180.0, 'limit_max': 180.0},
    'angle_C': {'limit_min': -180.0, 'limit_max': 180.0},
    'length_N': {'limit_min': 1.0, 'limit_max': 2.0},
    'length_A': {'limit_min': 1.0, 'limit_max': 2.0},
    'length_C': {'limit_min': 1.0, 'limit_max': 2.0},
}