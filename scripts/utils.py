def run_test(kernel, ts, nx, ny, nz, nt, target_dir, **kwargs):
  import os
  import subprocess
  from string import Template
  from scripts.utils import ensure_dir    
  from scripts.conf import conf

  job_template=Template(
"""$set_threads$th; $mpirun_cmd $pinning_cmd $pinning_args $exec_path --n-tests $ntests --disable-source-point --npx $npx --npy $npy --npz $npz --nx $nx --ny $ny --nz $nz  --verbose $verbose --target-ts $ts --nt $nt --target-kernel $kernel --cache-size $cs --thread-group-size $tgs --thx $thx --thy $thy --thz $thz --mwd-type $mwdt --num-wavefronts $nwf --t-dim $tb --threads $th_bind  --verify $verify | tee $outpath""")

  # set default arguments
  defaults = {'dry_run':0, 'is_dp':1, 'tgs':-1, 'cs':4096, 'mwdt':2, 'npx':1, 'npy':1, 'npz':1, 'nwf':-1,
              'ntests':2, 'alignment':16, 'verify':0, 'verbose':1, 'th': 1, 'thx': -1, 'thy': -1, 'thz': -1, 'tb':-1}
  # set the default run commands and arguments of the machine
  defaults.update(conf.machine_conf)

  # override the default arguments using the user specified ones
  defaults.update(kwargs)

  # create default file name if not set by the user:
  if 'outfile' not in kwargs.keys():
    defaults['outfile'] =  '_'.join(["%s_%s"%(k,v) for k,v in defaults.items()]).replace(" ","_").replace("-","_")

  # set the output path
  target_dir = os.path.join(os.path.abspath("."),target_dir)
  ensure_dir(target_dir)
  outpath = os.path.join(target_dir,defaults['outfile'])

  # set the executable
  if(defaults['is_dp']==1):
    exec_path = os.path.join(os.path.abspath("."),"build_dp/mwd_kernel")
  else:
    exec_path = os.path.join(os.path.abspath("."),"build/mwd_kernel")
  
  # set the processes number
  if defaults['mpirun_cmd'] != '':
    np = defaults['npx']*defaults['npy']*defaults['npz']
    defaults['mpirun_cmd'] = defaults['mpirun_cmd'] + " -np %d "%np

  # Disable manual binding if likwid is used
  if('likwid' in defaults['pinning_cmd']):
    defaults['th_bind'] = -1
  else:
    defaults['th_bind'] = defaults['th']

  job_cmd = job_template.substitute(nx=nx, ny=ny, nz=nz, nt=nt, kernel=kernel, ts=ts, outpath=outpath, 
                                    exec_path=exec_path, **defaults)
 
  print job_cmd
  if(defaults['dry_run']==0): sts = subprocess.call(job_cmd, shell=True)

  return job_cmd

def select_fields(data, rows=[], cols=[]):

  sel_data = []
  # select rows
  if rows:
    for k in data:
      for row in rows:
        cmp_res = [k[field] == val for field, val in row.iteritems()]
        if all(cmp_res):
          if cols:
            tup = {}
            for col in cols:
              tup[col] = k[col]
            sel_data.append(tup)
          else: # select all columns
            sel_data.append(dict(k))
        

  return sel_data


def parse_results(is_tgs_only=0):
  import os
  from csv import DictReader

  # parse existing results to aviod redundant autotuning time
  data_file = os.path.join('results', 'summary.csv')
  try:
    with open(data_file, 'rb') as output_file:
      data = list(DictReader(output_file))
  except:
    data = []

  # add fields to the entries
  mwdt_l = set()
  for k in data:
    k['stencil'] = get_stencil_num(k)
    k['method'] = 2 if 'Diamond' in k['Time stepper orig name'] else 0
    if k['method'] == 2:

      if k['Wavefront parallel strategy'] == 'Relaxed synchronization wavefront with fixed execution':
        k['mwdt'] = 3
      elif k['Wavefront parallel strategy'] == 'Relaxed synchronization wavefront':
        k['mwdt'] = 2
      elif k['Wavefront parallel strategy'] == 'Wavefront':
        k['mwdt'] = 0
      elif k['Wavefront parallel strategy'] == 'Fixed execution wavefronts':
        k['mwdt'] = 1
      if int(k['Thread group size']) == 1:
        k['mwdt'] = -1
      mwdt_l.add(k['mwdt'])

  params = dict()
  if(is_tgs_only==0):
    for k in data:
      try:
        if k['method']==2:
          if('tgs-1_' in k['file_name']): k['User set thread group size'] = -1
          if( (int(k['User set thread group size'])==-1) and  k['mwdt']==-1 ): # special case when autotuner selects 1WD 
            for m in mwdt_l:
              if m > 0:
                params[( m, k['stencil'], int(k['Global NX']), int(k['User set thread group size']), k['LIKWID performance counter'] )] = (int(k['Time unroll']), int(k['Multi-wavefront updates']), int(k['Thread group size']), int(k['Threads along x-axis']), int(k['Threads along y-axis']), int(k['Threads along z-axis']))
          else: # regular case
            params[( k['mwdt'], k['stencil'], int(k['Global NX']), int(k['User set thread group size']), k['LIKWID performance counter'] )] = (int(k['Time unroll']), int(k['Multi-wavefront updates']), int(k['Thread group size']), int(k['Threads along x-axis']), int(k['Threads along y-axis']), int(k['Threads along z-axis']))
      except:
        print k
        raise

  else: # TGS only parsing
    for k in data:
      try:
        if k['method']==2:
          params[( k['mwdt'], k['stencil'], int(k['Global NX']), int(k['User set thread group size']), k['LIKWID performance counter'] )] = (int(k['Time unroll']), int(k['Multi-wavefront updates']), int(k['Thread group size']), int(k['Threads along x-axis']), int(k['Threads along y-axis']), int(k['Threads along z-axis']))
      except:
        print k
        raise
  return params



def create_project_tarball(dest_dir, fname):
  import tarfile, glob
  import os
  from shutil import copyfile
  import inspect

  nl = ["src/kernels/*", "src/*.c", "src/*.h", "scripts/*.py", "scripts/*/*.py", "make.inc", "Makefile"]
  nl = [glob.glob(n) for n in nl]
  nl = [n for nn in nl for n in nn]

  out_dir = os.path.join(os.path.abspath("."),dest_dir)
  ensure_dir(out_dir)
  out_name = os.path.join(out_dir, fname+".tar.gz")

  print "Writing project files to:" + out_name
  with tarfile.open(out_name, "w:gz") as tar:
    for n in nl:
      print "Adding to the tar file: " + n
      tar.add(n)

  # copy the calling script to the destination dir
  src_file = inspect.stack()[1][1]
  dst_file = os.path.join(out_dir, os.path.basename(src_file))
  copyfile(src_file, dst_file)


def ensure_dir(d):
  import os, errno
  try:
    os.makedirs(d)
  except OSError as exc:
    if exc.errno == errno.EEXIST:
      pass
    else: raise

def load_csv(data_file):
  from csv import DictReader
  with open(data_file, 'rb') as output_file:
    data = DictReader(output_file)
    data = [k for k in data]
  return data

def get_stencil_num(k):
  # add the stencil operator
  if  'Solar' in k['Stencil Kernel coefficients']:
    return 6
  
  if  k['Stencil Kernel coefficients'] in 'constant':
    if  int(k['Stencil Kernel semi-bandwidth'])==4:
      stencil = 0
    else:
      stencil = 1
  elif  'no-symmetry' in k['Stencil Kernel coefficients']:
    stencil = 5
  elif  'sym' in k['Stencil Kernel coefficients']:
    if int(k['Stencil Kernel semi-bandwidth'])==1:
      stencil = 3
    else:
      stencil = 4
  else:
    stencil = 2
  return stencil


