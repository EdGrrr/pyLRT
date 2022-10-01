import numpy as np
import subprocess
import io
import os
import tempfile
import xarray as xr


class RadTran():
    '''The base class for handling a LibRadTran instance.
    First intialise and instance, then edit the properties using the 
    'options' directory. 

    Run the radiative transfer code by calling the 'run' method.

    Set the verbose option to retrieve the verbose output from UVSPEC.'''

    def __init__(self, folder):
        '''Create a radiative transfer object.
        folder - the folder where libradtran was compiled/installed'''
        self.folder = folder
        self.options = {}
        self.cloud = None

    def run(self, verbose=False, print_input=False, print_output=False, regrid=True, quiet=False):
        '''Run the radiative transfer code
        - verbose - retrieves the output from a verbose run, including atmospheric
                    structure and molecular absorption
        - print_input - print the input file used to run libradtran
        - print_output - echo the output
        - regrid - converts verbose output to the regrid/output grid, best to leave as True
        - quiet - if True, do not print UVSPEC warnings'''
        if self.cloud:  # Create cloud file
            tmpcloud = tempfile.NamedTemporaryFile(delete=False)
            cloudstr = '\n'.join([
                ' {:4.2f} {:4.2f} {:4.2f}'.format(
                    self.cloud['z'][alt],
                    self.cloud['lwc'][alt],
                    self.cloud['re'][alt])
                for alt in range(len(self.cloud['lwc']))])
            tmpcloud.write(cloudstr.encode('ascii'))
            tmpcloud.close()
            self.options['wc_file 1D'] = tmpcloud.name

        if verbose:
            try:
                del(self.options['quiet'])
            except:
                pass
            self.options['verbose'] = ''

        inputstr = '\n'.join(['{} {}'.format(name, self.options[name])
                              for name in self.options.keys()])
        if print_input:
            print(inputstr)
            if self.cloud:
                print('Cloud')
                print('  Alt  LWC   Re')
                print(cloudstr)
            print('')

        cwd = os.getcwd()
        os.chdir(os.path.join(self.folder, 'bin'))

        process = subprocess.run([os.getcwd()+'/uvspec'], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 input=inputstr, encoding='ascii')
        os.chdir(cwd)

        if self.cloud:
            os.remove(tmpcloud.name)
            del(self.options['wc_file 1D'])

        # Check uvspec output for errors/warnings
        if not quiet:
            print_flag = False
            for line in io.StringIO(process.stderr):
                if line.startswith('*** Warning'):
                    # Some uvspec warnings start with three stars
                    # These have three stars for every line
                    print(line.strip())
                elif line.startswith('*****'):
                    # Many uvspec warnings are in a star box
                    print(line.strip())
                    print_flag = not(print_flag)
                elif print_flag:
                    # Print line if we are within a star box
                    print(line.strip())

        #Check for errors!
        error = ['UVSpec Error Message:\n']
        error_flag = False
        for line in io.StringIO(process.stderr):
            if line.startswith('Error'):
                error_flag = True
            if error_flag:
                error.append(line)
        if error_flag:
            error = ''.join(error)
            raise ValueError(error)

        if print_output:
            print('Output file:')
            print(process.stdout)
        if verbose:
            try:
                del(self.options['verbose'])
            except:
                pass
            self.options['quiet'] = ''

            return (np.genfromtxt(io.StringIO(process.stdout)),
                    _read_verbose(io.StringIO(process.stderr), regrid=regrid))
        return np.genfromtxt(io.StringIO(process.stdout))


def _skiplines(f, n):
    '''Skip n lines from file f'''
    for i in range(n):
        _ = f.readline()


def _skiplines_title(f, n, t):
    '''Skip n lines from file f. Return the title on line t'''
    for i in range(n):
        if i == t:
            title = [a.strip() for a in f.readline().strip().split('|')]
        _ = f.readline()
    return title


def _match_table(f, start_idstr, nheader_rows, header_row=None):
    '''Get the data from an individual 2D table. 
    start_idstr - string to locate table
    nheader_rows - number of rows to skip before the data'''
    while True:
        line = f.readline()
        if line.startswith(start_idstr):
            break
    title = _skiplines_title(f, nheader_rows, header_row)
    profiles = []
    while True:
        line = f.readline()
        if line.startswith(' --'):
            break
        elif line.startswith('T'):
            break
        else:
            profiles.append(line.replace('|', ''))
    profiles = np.genfromtxt(io.StringIO(''.join(profiles)))
    return title, profiles


def _read_table(f, start_idstr, labels, wavelengths, regrid=False):
    '''Read in the 3D data tables (e.g. optical properties)'''
    optprop = []
    num_wvl = len(wavelengths['wvl'])
    for wv in range(num_wvl):
        temp = _match_table(f, start_idstr, 4, 2)
        optprop.append(temp[1])
        # Could potentially read variable names from the table in future
        #optproplabels = temp[0]
    optprop = np.array(optprop)
    if regrid:
        optprop = _map_to_outputwvl(optprop, wavelengths)
        optprop = xr.Dataset(
            {labels[a]: (['wvl', 'lc'], optprop[:, :, a])
             for a in range(1, optprop.shape[2])},
            coords={'lc': range(optprop.shape[1]),
                    'wvl': np.unique(wavelengths['OutputWVL'].data)})
    else:
        optprop = xr.Dataset(
            {labels[a]: (['wvl', 'lc'], optprop[:, :, a])
             for a in range(1, optprop.shape[2])},
            coords={'lc': range(optprop.shape[1]),
                    'wvl': wavelengths['wvl']})
    return optprop


def _get_wavelengths(f):
    '''Readin the wavelength information'''
    while True:
        line = f.readline()
        if line.startswith(' ... calling setup_rte_wlgrid()'):
            number = int(f.readline().strip().split(' ')[0])
            break
    _skiplines(f, 1)
    wavelengths = []
    for i in range(number):
        wavelengths.append([float(a.strip().replace(' nm', ''))
                            for a in f.readline().strip().split('|')])
    wavelengths = np.array(wavelengths)
    return xr.Dataset(
        {'OutputWVL': (['wvl'], wavelengths[:, 0]),
         'Weights': (['wvl'], wavelengths[:, 2])},
        coords={'wvl': wavelengths[:, 1]})


def _read_verbose(f, regrid=False):
    '''Readin the uotput from 'verbose' to a set of xarrays'''
    try:
        wavelengths = _get_wavelengths(f)
    except:
        print('Readin of verbose file failed.')
        for i in range(10):
            print(f.readline())
        return None

    profiles = _match_table(f, '*** Scaling profiles', 4, 1)
    proflabels = ['lc', 'z', 'p', 'T', 'air', 'o3',
                  'o2', 'h2o', 'co2', 'no2', 'o4']
    profiles = xr.Dataset(
        {proflabels[a]: (['lc'], profiles[1][:, a])
         for a in range(1, profiles[1].shape[1])},
        coords={'lc': range(profiles[1].shape[0])})

    gaslabels = ['lc', 'z', 'rayleigh_dtau', 'mol_abs',
                 'o3', 'o2', 'h2o', 'co2', 'no2', 'bro', 'oclo',
                 'hcho', 'o4', 'so2', 'ch4', 'n2o', 'co', 'n2']

    redistlabels = ['lc', 'z',
                    'o3', 'o2', 'co2', 'no2', 'bro',
                    'oclo', 'hcho', 'wc.dtau', 'ic.dtau']

    optproplabels = ['lc', 'z', 'rayleigh_dtau',
                     'aer_sca', 'aer_abs', 'aer_asy',
                     'wc_sca', 'wc_abs', 'wc_asy',
                     'ic_sca', 'ic_abs', 'ic_asy',
                     'ff', 'g1', 'g2', 'f', 'mol_abs']

    gases = _read_table(f, '*** setup_gases', gaslabels, wavelengths, regrid=regrid)
    redist = _read_table(f, '*** setup_redistribute', redistlabels, wavelengths, regrid=regrid)
    optprop = _read_table(f, '*** optical_properties',
                          optproplabels, wavelengths, regrid=regrid)

    return {'wavelengths': wavelengths,
            'profiles': profiles,
            'gases': gases,
            'redist': redist,
            'optprop': optprop}


def _map_to_outputwvl(data, wavelengths):
    '''Regrids the table properties to the output wavelengths
    
    NOTE: data is still a numpy array'''
    output_wvl = np.array(np.unique(wavelengths['OutputWVL'].data))
    if len(data.shape)>1:
        opdata = np.zeros([output_wvl.shape[0]] + list(data.shape)[1:])
    else:
        opdata = np.zeros(output_wvl.shape[0])
    for i in range(wavelengths['wvl'].shape[0]):
        opdata[np.where(output_wvl==wavelengths['OutputWVL'].data[i])[0]] += wavelengths['Weights'].data[i]*data[i]
    return opdata


