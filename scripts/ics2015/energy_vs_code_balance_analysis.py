#!/usr/bin/env python
def main():
  import sys

  raw_data = load_csv(sys.argv[1])

  k_l = set()
  for k in raw_data:
    k_l.add(get_stencil_num(k))
  k_l = list(k_l)

  n_l = set()
  for k in raw_data:
    n_l.add(k['Global NX'])
  n_l = list(n_l)


  for k in k_l:
    for N in n_l:
      gen_res(raw_data, int(k), int(N))

#    gen_res(raw_data)


def gen_res(raw_data, stencil_kernel, N):
  from operator import itemgetter
  import matplotlib.pyplot as plt
  import pylab
  from csv import DictWriter
  from operator import itemgetter


  #fig_width = 8.588*0.393701 # inches
  fig_width = 6.0*0.393701 # inches
  fig_height = 0.68*fig_width #* 210.0/280.0#433.62/578.16

  fig_size =  [fig_width,fig_height]
  params = {
         'axes.labelsize': 6,
         'axes.linewidth': 0.5,
         'lines.linewidth': 0.75,
         'text.fontsize': 7,
         'legend.fontsize': 5,
         'xtick.labelsize': 6,
         'ytick.labelsize': 6,
         'lines.markersize': 3,
         'font.size': 6,
         'text.usetex': True,
         'figure.figsize': fig_size}
  pylab.rcParams.update(params)


  req_fields = [('Total cache block size (kiB)', int), ('MStencil/s  MAX', float), ('Time stepper orig name', str), ('Stencil Kernel semi-bandwidth', int), ('Stencil Kernel coefficients', str), ('Precision', str), ('Time unroll',int), ('Number of time steps',int), ('Number of tests',int), ('Local NX',int), ('Local NY',int), ('Local NZ',int), ('Total Memory Transfer', float), ('Thread group size' ,int), ('Intra-diamond prologue/epilogue MStencils',int), ('Multi-wavefront updates', int), ('Intra-diamond width', int), ('Energy', float), ('Energy DRAM', float)]
  data = []
  for k in raw_data:
    tup = dict()
    # add the general fileds
    for f in req_fields:
      try:
        tup[f[0]] = map(f[1], [k[f[0]]] )[0]
      except:
        pass
#        print f[0]
    # add the stencil operator
    tup['Kernel'] = get_stencil_num(k)
    data.append(tup)


  data1 = []
  dd1 = data[:]
  dd2 = data[:]
  for d1 in dd1:
    for d2 in dd2:
      if d1['Time unroll'] == d2['Time unroll'] and d1['Kernel'] == d2['Kernel']:
        if 'Energy' in d1 and 'Energy' not in d2:
          d1['Total Memory Transfer'] = d2['Total Memory Transfer']
          data1.append(d1)

  #for i in data1: 
  #  print i['Kernel'],i['Time unroll'],i['Energy'],i['Total Memory Transfer'],i['Energy DRAM']
  #print ""


  WS = 8 # word size in bytes
  data2 = []
  for tup in data1:
    tup['Actual Bytes/LUP'] = actual_BpU(tup)
    tup['Model'] = models(tup)
    # model error
    tup['Err %'] = 100 * (tup['Model'] - tup['Actual Bytes/LUP'])/tup['Actual Bytes/LUP']
    tup['D_width'] = tup['Intra-diamond width']
    tup['Performance'] = tup['MStencil/s  MAX']
 
    # energy

    ext_lups = tup['Intra-diamond prologue/epilogue MStencils']*1e6
    NT = tup['Local NX']**3 * tup['Number of time steps']
    Neff = NT - ext_lups
    lups = tup['Number of tests'] * Neff / 1e9
    tup['cpu energy'] = tup['Energy']/lups
    tup['dram energy'] = tup['Energy DRAM']/lups
    tup['total energy'] = tup['dram energy'] + tup['cpu energy']

#   tup['Cache block'] = get_bs(Dw=tup['D_width'], Nd=get_nd(tup['Kernel']), Nf=(tup['Multi-wavefront updates']-1), Nx=tup['Local NX'], WS=WS)  
    data2.append(tup)
#    try: print "%6.3f  %6.3f  %6.3f" % (tup['Cache block'], tup['Total cache block size (kiB)']/1024.0,tup['Cache block']- tup['Total cache block size (kiB)']/1024.0)
#    except: pass

  #for i in data2: print i
  data2 = sorted(data2, key=itemgetter('Kernel', 'Local NX', 'D_width'))


  et=[]
  ec=[]
  er=[]
  cb=[]
  cb_meas=[]
  Dw=[]
  perf = []
  for k in data2:
    if k['Kernel']==stencil_kernel and k['Local NX']==N:
      et.append(k['total energy'])
      ec.append(k['cpu energy'])
      er.append(k['dram energy'])
      cb.append(k['Model'])
      cb_meas.append(k['Actual Bytes/LUP'])
      Dw.append(k['D_width'])
      perf.append("%.2f" % (k['Performance']/1000.0))

#  for i in range(len(et)):
#    print Dw[i], et[i], cb_meas[i], cb[i], perf[i]

  if Dw==[]: return

  fig, ax = plt.subplots()
  #ax.plot(cb,      cs, marker='^', linestyle='-', color='k', label="Model")
  ax.plot(cb_meas, et, marker='*', linestyle='--', color='k', label="Total")
  ax.plot(cb_meas, ec, marker='x', linestyle='--', color='m', label="CPU")
  ax.plot(cb_meas, er, marker='+', linestyle='--', color='b', label="DRAM")
  ax.set_xlabel('Measured code balance (Bytes/LUP)')
  ax.set_ylabel('pJ/LUP')
  ax.set_xlim([0, max(cb_meas+cb)+1])
  ax.set_ylim([0, max(et)+0.5])
  ax2 = ax.twiny()
  ax2.set_xticks(cb_meas)
  ax2.set_xlabel('GLUP/s')
  ax2.set_xticklabels(perf, rotation=60)
  ax2.set_xlim(ax.get_xlim())

  for i, d in enumerate(Dw):
    #if ((d+4)%8 == 0):
    ys = -max(et)/11
    xs = 0
    an = d
    if d==8: 
      an="Dw=8"
      #xs = -cb_meas[0]/9
    if stencil_kernel==5 and d==4:
      xs = -1
    ax.annotate(an, (cb_meas[i]+xs, et[i]+ys))  

  title = '_code_balance_vs_energy_N'+str(N)
  if stencil_kernel == 0:
      title = '25_pt_const' + title
  elif stencil_kernel == 1:
      title = '7_pt_const' + title
  elif stencil_kernel == 4:
      title = '25_pt_var' + title
  elif stencil_kernel == 5:
      title = '7_pt_var' + title

  ax.legend(loc='best')
#  ax.grid()
#  pylab.savefig(title+'.png', bbox_inches="tight", pad_inches=0.04)
  pylab.savefig(title+'.pdf', format='pdf', bbox_inches="tight", pad_inches=0)
  plt.clf()



def actual_BpU(tup):
  total_mem = tup['Total Memory Transfer']
  R = tup['Stencil Kernel semi-bandwidth']
  nt = tup['Number of time steps'] * tup['Number of tests']

  nx = tup['Local NX']
  ny = tup['Local NY']
  nz = tup['Local NZ']

  oh = tup['Intra-diamond prologue/epilogue MStencils']

  stencil_size = 2*ny*nz + ny*nz*(nx+2*R) 
  BpU = (total_mem * 10**9) / ( stencil_size * nt - oh*10**6*tup['Number of tests'])

  #print BpU, total_mem, stencil_size, nt, oh, tup['Number of tests']
  return BpU


def models(tup):
  if   tup['Precision'] == 'DP': word_size = 8
  elif tup['Precision'] == 'SP': word_size = 4

  R = tup['Stencil Kernel semi-bandwidth']
  TB = tup['Time unroll']
  ny = tup['Local NY']

  # number of streamed copies of the domain (buffers)
  nb = get_nd(tup['Kernel'])

  width = tup['Intra-diamond width']
  YT_section = float((TB+1)**2 * 2 * R)

  # temporal blocking model
  if tup['Time stepper orig name'] == 'Spatial Blocking':
    bpu = (1 + nb) * word_size
  else: # no temporal blocking model
    bpu = ( ((width - 2*R) + width) + (nb*width + 2*R) ) * word_size / YT_section
 
  return bpu


def get_bs(Dw, Nd, Nf, Nx, WS):
  Ww = Dw + Nf - 1
  Bs = WS*Nx*( Nd*(Dw**2/2.0 + Dw*Nf) + 2.0*(Dw+Ww) )
  if Dw==0: Bs=0
  return Bs/(1024.0*1024.0)


def get_nd(k):
  if k== 0:
    nd = 2 + 1
  if k== 1:
    nd = 2
  if k== 4:
    nd = 2 + 13
  if k== 5:
    nd = 2 + 7
  
  return nd


def get_stencil_num(k):
  # add the stencil operator
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


def load_csv(data_file):
  from csv import DictReader
  with open(data_file, 'rb') as output_file:
    data = DictReader(output_file)
    data = [k for k in data]
  return data
    
    
if __name__ == "__main__":
  main()

