import pyLRT
import numpy as np
import pytest
import unittest

LIBRADTRAN_FOLDER = pyLRT.get_lrt_folder()

# Ability to parse the test input and output files for libradtran
def parse_inputfile(inputdata):
    output_dict = {}
    while True:
        line = inputdata.readline()
        if len(line) == 0:
            break
        if line.startswith('#'):
            continue
        
        line = line.strip()

        # Remove comments
        if line.find('#') >= 0:
            # There is a comment - remove it
            line = line[:line.find('#')].strip()

        if len(line) == 0:
            continue

        line = line.replace('../', LIBRADTRAN_FOLDER)
        # Deal with includes (todo)
        if line.startswith('include'):
            with open(line.split()[1]) as f:
                newdict = parse_inputfile(f)
            output_dict.update(newdict)
            continue
            # for name in newdict.keys():
            #     output_dict[name] = newdict[name]

        tokens = line.split()
        if tokens[0] in ['mol_modify',
                         'aerosol_modify', 'aerosol_file',
                         'wc_modify', 'wc_file',
                         'profile_properties', 'profile_file',
                         'sslidar',
                         'cloudcover']:
            key, value = ' '.join(tokens[:2]), ' '.join(tokens[2:])
        elif len(tokens) == 1:
            key, value = tokens[0], ''
        else:
            key, value = tokens[0], ' '.join(tokens[1:])
        output_dict[key] = value
    return output_dict

def parse_outputfile(parser, filename, rt):
    return parser.parse_output(np.genfromtxt(filename), rt)

def uvspec_test_function(uvspec_testdata):
    test_name, output_name, atol, rtol = uvspec_testdata
    
    test_filename = f'{LIBRADTRAN_FOLDER}/examples/{test_name}.INP'
    
    with open(test_filename) as f:
        test_dict = parse_inputfile(f)
        lrt = pyLRT.RadTran(LIBRADTRAN_FOLDER)
        for name in test_dict.keys():
            lrt.options[name] = test_dict[name]

    sdata = lrt.run(parse=True)

    if output_name:
        output_filename = f'{LIBRADTRAN_FOLDER}/examples/{output_name}.OUT'
    else:
        output_filename = test_filename.replace('.INP', '.OUT')
    testdata = parse_outputfile(lrt.parser, output_filename, rt=lrt)

    abs_error = ((sdata-testdata)).apply(np.abs)
    rel_error_abs = (abs_error/testdata).apply(np.abs)

    rel_error_flag = (((rel_error_abs>rtol)*(abs_error>atol)).sum()>0)

    for name in list(rel_error_flag.keys()):
         assert rel_error_flag[name] == False


@pytest.mark.parametrize("uvspec_testdata", [
    ('UVSPEC_CLEAR', '', 0.00001,0.001),
    ('UVSPEC_AVHRR_SOLAR_CH1', '', 0.00001,0.001),
    ('UVSPEC_AVHRR_SOLAR_CH2', '', 0.001,0.001),
    ('UVSPEC_AVHRR_SOLAR_CH3', '', 0.00001,0.001),
    ('UVSPEC_AVHRR_THERMAL_CH3', '', 0.00001,0.001),
    ('UVSPEC_AVHRR_THERMAL_CH4', '', 0.00001,0.001),
    ('UVSPEC_AVHRR_THERMAL_CH5', '', 0.00001,0.001),
    ('UVSPEC_AEROSOL_MOMENTS', '', 0.00001,0.001),
    ('UVSPEC_AEROSOL_REFRAC', '', 0.00001,0.001),
    ('UVSPEC_AEROSOL', '', 0.00001,0.001),
    ("UVSPEC_CLEAR", '', 0.00001,0.001),
    ("UVSPEC_CLOUDCOVER_REDISTRIBUTION", "UVSPEC_CLOUDCOVER", 0.001  ,0.018),
    ("UVSPEC_CLOUDCOVER", '', 0.001  ,0.018),
    ("UVSPEC_DISORT", '', 0.00001,0.001),
    ("UVSPEC_TWOSTR", '', 0.00001,0.001),
    ("UVSPEC_FLUORESCENCE", '', 0.1,0.001),
    ("UVSPEC_SO2", '', 0.00001,0.001),
    ("UVSPEC_TRANSMITTANCE_WL_FILE", '', 0.001, 0.001),
    ("UVSPEC_TWOMAXRND3C", '', 0.00001, 0.001),
    ("UVSPEC_WC_IC_IPA_FILES", '', 0.00001, 0.001),
    ("UVSPEC_WC_MOMENTS", '', 0.00001, 0.001),
    ("UVSPEC_WC", '', 0.00001, 0.001),
])
def test_uvspec(uvspec_testdata):
    uvspec_test_function(uvspec_testdata)

@pytest.mark.xfail
@pytest.mark.parametrize("uvspec_testdata", [
    ("UVSPEC_PROFILES1", '', 0.00001,0.001),
    ("UVSPEC_PROFILES2", '', 0.00001,0.001),
    ("UVSPEC_PROFILES3", '', 0.00001,0.001),
    ("UVSPEC_PROFILES4", '', 0.00001,0.001),
    ("UVSPEC_REPINT_THERMAL", '', 0.01, 0.001), # Has zout in output
    ("UVSPEC_COOLING_IPA", '', 0.001, 0.019), # Requires zout
    ("UVSPEC_HEATING_IPA", '', 0.001  ,0.0017), # Requires zout
])
def test_uvspec_zout(uvspec_testdata):
    uvspec_test_function(uvspec_testdata)

@pytest.mark.xfail
@pytest.mark.parametrize("uvspec_testdata", [
    ("UVSPEC_RADIANCES", '', 0.00001, 0.001), # Fails due to requiring uu output
    ("UVSPEC_BRDF_AMBRALS_FILE", '', 0.0005 ,0.001), # Requires uu
])
def test_uvspec_uu(uvspec_testdata):
    uvspec_test_function(uvspec_testdata)
    
@pytest.mark.xfail
@pytest.mark.parametrize("uvspec_testdata", [
    ("UVSPEC_RODENTS", '', 0.00001, 0.001), # Rodents solver
    ("tests/UVSPEC_RGB", '', 0.00001, 0.001), # Output process RGB
    ("UVSPEC_SSLIDAR", '', 0.00001,0.001), # Requires lidar solver
])
def test_uvspec_non_disort(uvspec_testdata):
    uvspec_test_function(uvspec_testdata)


