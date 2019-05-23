#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import argparse




parser = argparse.ArgumentParser( description = 
"""The program will anticipate 2 inputs from user.  These 2 inputs will specify the range of ZINC ID to fetch.  Smaller ID is the first input, and the larger ID is the second input.""", 
                                  formatter_class = argparse.RawTextHelpFormatter)

parser.add_argument( 'rng',
                     metavar = 'int1 int2 -- Range of ZINC ID',
                     type    = int,
                     nargs   = 2)

args = parser.parse_args()




def fetchZinc(id):
    f = 'https://zinc15.docking.org/substances/ZINC%d' % id
    r = requests.get(f, timeout = 10) # ...timeout after 10 sec

    v = {}

    # Fetch data only if web page responds correctly...
    if r.status_code == 200: 
        c = r.content
        s = BeautifulSoup(c, features='lxml')
        t = s.findAll('table') # ...All tables

        v['ID']   = id
        v['MW']   = float(t[0].findAll('td')[3].string)
        v['logP'] = float(t[0].findAll('td')[4].string)
        v['Ring'] =   int(t[1].findAll('td')[1].string)
        v['HBD']  =   int(t[2].find('td', {'title':"Hydrogen Bond Donors"}).string)
        v['HBA']  =   int(t[2].find('td', {'title':"Hydrogen Bond Acceptors"}).string)
        v['RB']   =   int(t[2].find('td', {'title':"Rotatable Bonds"}).string)

    return v




rng = args.rng  # ...range of IDs
fo   = 'ZINC_%d.dat' % rng[0]
with open(fo,'w') as fh:
    fh.write("%11s  %-8s  %s  %s  %s  %s  %s\n" % ( 'ID',  
                                                    'MW',  
                                                    'logP',
                                                    'Ring',
                                                    'HBD', 
                                                    'HBA',  
                                                    'RB' ))


    v = {}
    for i in range(rng[0],rng[-1]):
        f = 'ZINC_%d' % i
        try: 
            v = fetchZinc(i)

            fh.write("%11d  %6.2f  %6.2f  %3d  %3d  %3d  %3d\n" % ( v['ID'],   
                                                                    v['MW'],   
                                                                    v['logP'], 
                                                                    v['Ring'], 
                                                                    v['HBD'] , 
                                                                    v['HBA'] , 
                                                                    v['RB']  ))
            fh.flush()

            print "%s is export.  " % f
        except: 
            print "%s is ignored.  " % f
            pass
