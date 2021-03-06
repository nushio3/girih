#!/usr/bin/env python
def igs_test(target_dir, exp_name, th, group='', params={}, dry_run=0, is_tgs_only=0): 
  from scripts.conf.conf import machine_conf, machine_info
  from scripts.utils import run_test
  import itertools

  is_dp=1

  cs = 4096

  th = machine_info['n_cores']
  k_r = {0:4, 1:1, 4:4, 5:1, 6:1}
  kernels_min_limits = {n: k_r[n]*th*4 for n in k_r.keys()}

  # Test using rasonable time
  # T = scale * size / perf
  # scale = T*perf/size
  desired_time = 20
  if(machine_info['hostname']=='Haswell_18core'):
    tgs_l = [2,3,6,9,18]
    increment = 128
    k_perf_order = {0:1500, 1:5000, 4:400, 5:2000 ,6:100}
    if is_dp == 1:
      kernels_limits = {0:1281, 1:1409, 4:769, 5:897, 6:513}
    else:
      kernels_limits = {0:2100, 4:1200}

  elif(machine_info['hostname']=='IVB_10core'):
    tgs_l = [2,5,10]
    increment = 64
    k_perf_order = {0:1200, 1:3000, 4:350, 5:1500 ,6:80}
    if is_dp ==1:
      kernels_limits = {0:1089, 1:1217, 4:577, 5:705, 6:289}
    else:
      kernels_limits = {0:1350, 4:801}


  points = dict()
  points[0] = [64] + list(range(128, 5000, increment)) 
  points[1] = points[0]
  points[4] = points[0]
  points[5] = points[0]
  points[6] = points[0]


  k_time_scale = {n: desired_time*k_perf_order[n] for n in k_perf_order.keys()}

  if(is_tgs_only):
    exp_l = [(2,tgs_l)]
  else:
    exp_l = [(2,[-1, 1]), (0,[0])]

  count=0
  for ts, tgs_rl in exp_l:
    for tgs_r in tgs_rl:
      for kernel, mwdt_list in [(0,[1]), (1,[2]), (4,[1]), (5,[2])]:
#      for kernel, mwdt_list in [(6,[1])]:
        if ts==0 or tgs_r==1:
          mwdt_list=[-1]
        for mwdt in mwdt_list:
          for N in points[kernel]:
            if( ((tgs_r!=1) or  (N >= kernels_min_limits[kernel])) and (N < kernels_limits[kernel]) ):
              tb, nwf, tgs, thx, thy, thz = (-1,-1,tgs_r,-1,-1,-1)
              key = (mwdt, kernel, N, tgs_r, group)
              if key in params.keys():
                print "Already computed:", key
                continue #already computed

              key = (mwdt, kernel, N, tgs_r, 'MEM')
              if key in params.keys(): #reuse the conf of existing test
                tb, nwf, tgs, thx, thy, thz = params[key]
              outfile=('kernel%d_isdp%d_ts%d_mwdt%d_tgs%d_N%d_%s_%s.txt' % (kernel, is_dp, ts, mwdt, tgs_r, N, group, exp_name[-13:]))
              nt = max(int(k_time_scale[kernel]/(N**3/1e6)), 30)
              c_mwdt = mwdt # to avoid negative array access at the c code
              if mwdt==-1: c_mwdt=1
#              print outfile, tb, nwf, tgs, thx, thy, thz
              run_test(dry_run=dry_run, is_dp=is_dp, th=th, tgs=tgs, thx=thx, thy=thy, thz=thz, kernel=kernel, ts=ts, nx=N, ny=N, nz=N, nt=nt, outfile=outfile, target_dir=target_dir, cs=cs, mwdt=c_mwdt, tb=tb, nwf=nwf)
              count = count+1
  return count

def main():
  from scripts.utils import create_project_tarball, get_stencil_num, parse_results
  from scripts.conf.conf import machine_conf, machine_info
  import os, sys
  import time,datetime

  # user params
  dry_run = 1   if len(sys.argv)<2 else int(sys.argv[1]) # dry run
  is_tgs_only=0
  if(len(sys.argv)>=3):
    if(sys.argv[2]=='tgs'): # whether to test all TGS combinations
      is_tgs_only=1

  sockets=1 # number of processors to use in the experiments

  time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H_%M')
  tgs_s = '' if is_tgs_only==0 else '_tgs'
  exp_name = "increasing_grid_size_sockets_%d%s_at_%s_%s" % (sockets,tgs_s,machine_info['hostname'], time_stamp)  

  tarball_dir='results/'+exp_name
  if(dry_run==0): create_project_tarball(tarball_dir, "project_"+exp_name)
  target_dir='results/' + exp_name 

  # parse the results to obtain the selected parameters by the auto tuner
  params = parse_results(is_tgs_only=is_tgs_only)


  #update the pinning information to use all cores
  th = machine_info['n_cores']*sockets

  if sockets == 1:
    pin_str = "S0:0-%d "%(th-1)
  if sockets == 2:
    pin_str = "S0:0-%d@S1:0-%d -i "%(th/2-1, th/2-1)

  count=0
#  for group in ['MEM']:
#  for group in ['MEM', 'L2', 'L3', 'TLB_DATA', 'DATA', 'ENERGY']:
  for group in ['DATA']:
#  for group in ['MEM', 'ENERGY', 'L2', 'L3', 'DATA']:
    if( (machine_info['hostname']=='IVB_10core') and (group=='TLB_DATA') ): group='TLB'
    machine_conf['pinning_args'] = "-m -g " + group + " -C " + pin_str + ' -s 0x03 --'

    count= count + igs_test(target_dir, exp_name, th=th, params=params, group=group, dry_run=dry_run,is_tgs_only=is_tgs_only) 

  print "experiments count =" + str(count)


if __name__ == "__main__":
  main()
