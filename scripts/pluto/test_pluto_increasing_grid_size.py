#!/usr/bin/env python

def igs_test(dry_run, target_dir, exp_name, group='', param_l=[]):
  from scripts.conf.conf import machine_info
  from scripts.pluto.pluto_utils import run_pluto_test
  import itertools, os, pickle
  from os.path import join as jpath

  target_dir = jpath(os.path.abspath("."),target_dir)


  # machine_info['hostname']=='IVB_10core'
  kernels_limits = {'3d25pt':1089, '3d7pt':1217, '3d25pt_var':577, '3d7pt_var':769}
  increment = 64

  if(machine_info['hostname']=='Haswell_18core'):
    kernels_limits = {'3d25pt':1281, '3d7pt':1409, '3d25pt_var':769, '3d7pt_var':897}
    increment = 128

  points = dict()
  points['3d7pt'] = [64] + list(range(128, 5000, increment))
  points['3d7pt_var'] = points['3d7pt']
  points['3d25pt'] = points['3d7pt']
  points['3d25pt_var'] = points['3d7pt']

  count=0
  #for kernel in ['3d7pt', '3d7pt_var', '3d25pt']:#, '3d25pt_var']:
  for kernel in [ '3d25pt', '3d25pt_var']:
    for N in points[kernel]:
      if (N < kernels_limits[kernel]):
        outfile=('pluto_kernel_%s_N%d_%s_%s.txt' % (kernel, N, group, exp_name[-13:]))
        outfile = jpath(target_dir, outfile)
        if(dry_run==1):
          nt=32; param=[-1,-1,-1]
#        nt = max(int(k_time_scale[kernel]/(N**3/1e6)), 30)
        if (kernel, N, group) in param_l.keys():
          continue # results exist for this test case

        if (kernel, N, 'MEM') in param_l.keys(): # use the tuned params of memory results
          if(dry_run==0): fp = open(outfile, 'w')
          param, nt = param_l[(kernel, N, 'MEM')]
          nt = nt*2
        else:
#          continue
          if(dry_run==0): 
            fp = open(outfile, 'w')
            param, nt, tune_res = pluto_tuner(kernel=kernel, nx=N, ny=N, nz=N, fp=fp)
            with open(outfile[:-3]+'p', 'w') as fpickle:
              pickle.dump(tune_res, fpickle)
        if(dry_run==0): tee(fp, outfile)
#        print outfile, param
        test_str, telapsed = run_pluto_test(dry_run=dry_run, kernel=kernel, nx=N, ny=N, nz=N, nt=nt, params=param, outfile=outfile)
        if(dry_run==0):
          tee(fp, test_str)
          fp.close()
        count = count+1
  return count

def main():
  from scripts.utils import create_project_tarball, get_stencil_num
  from scripts.conf.conf import machine_conf, machine_info
  import os, sys
  from csv import DictReader
  import time,datetime

  dry_run = 1 if len(sys.argv)<2 else int(sys.argv[1])

  time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H_%M')
  exp_name = "pluto_increasing_grid_size_at_%s_%s" % (machine_info['hostname'], time_stamp)

  tarball_dir='results/'+exp_name
  if(dry_run==0): create_project_tarball(tarball_dir, "test_"+exp_name)
  target_dir='results/' + exp_name

  # parse the results to find out which of the already exist
  data = []
  data_file = os.path.join('results', 'summary.csv')
  try:
    with open(data_file, 'rb') as output_file:
      raw_data = DictReader(output_file)
      for k in raw_data:
        kernel = get_stencil_num(k)
        if(kernel==0):
          k['stencil'] ='3d25pt'
        elif(kernel==1):
          k['stencil'] ='3d7pt'
        elif(kernel==4):
          k['stencil'] ='3d25pt_var'
        elif(kernel==5):
          k['stencil'] ='3d7pt_var'
        else:
          raise
        data.append(k)
  except:
     pass
  param_l = dict()
  for k in data:
    try:
      param_l[(k['stencil'], int(k['Global NX']), k['LIKWID performance counter']  ) ] = ([int(k['PLUTO tile size of loop 1']), int(k['PLUTO tile size of loop 3']), int(k['PLUTO tile size of loop 4'])], int(k['Number of time steps']) )
    except:
      print k
      raise

  #update the pinning information to use all cores
  th = machine_info['n_cores']

  pin_str = "0-%d "%(th-1)

  count = 0
  for group in ['MEM']:
#  for group in ['MEM', 'L2', 'L3', 'DATA', 'TLB_DATA', 'ENERGY']:
    if(machine_info['hostname']=='Haswell_18core'):
      machine_conf['pinning_args'] = " -m -g " + group + " -C S1:" + pin_str
    elif(machine_info['hostname']=='IVB_10core'):
      if group=='TLB_DATA': group='TLB'
      machine_conf['pinning_args'] = " -g " + group + " -C S0:" + pin_str
#    for k,v in param_l.iteritems(): print k,v
    count = count + igs_test(dry_run, target_dir, exp_name, param_l=param_l, group=group)

  print "experiments count =" + str(count)


if __name__ == "__main__":
  main()
