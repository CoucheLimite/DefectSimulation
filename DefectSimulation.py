import numpy as np
import scipy.constants as const
from scipy.integrate import odeint
from semiconductor.material.thermal_velocity import ThermalVelocity as Vel_th
from semiconductor.material.intrinsic_carrier_density import IntrinsicCarrierDensity as NI
from semiconductor.electrical.mobility import Mobility as mob
from semiconductor.electrical.ionisation import Ionisation as Ion
import warnings
import matplotlib.pyplot as plt


kb = const.k / const.e

class defectSimu():
    '''
    In this code, add a defet by definding its charge states, energy level Et
    and capture cross sections (sigma_e and sigma_h) for transition between two
    charge states and the total concentration Nt.
    The order of the energy level, capture cross section and the occupany f
    calculated are from the most negatively charged states to the most postively
    charged states. See below:
    Donor Type D: Et = E0/+, f = [f0, f+, 0] The 0 here is meaningless, just to
    make the list length the same as two level defect
    Acceptor Type A: Et = E-/0, f = [f-, f0, 0] The 0 here is meaningless
    Donor-acceptor Type AD: Et = [E-/0, E0/+], f = [f-, f0, f+]
    Double-acceptor AA: Et = [E--/-, E-/0], f = [f--, f-, f0]
    Double-donor Type DD: Et = [E0/+, E+/++], f = [f0, f+, f++]
    '''
    def __init__(self, Ndop=1e16, doptype='p', temp=300, tol=1e-10, maxite = 5000, dampfactor=1e-2, **kwarg):
        '''
        Initiate the class with the sample parameter
        '''
        self.Ndop = Ndop
        self.doptype = doptype
        self.temp = temp
        self.defect_list=[]
        self.tol = tol
        self.dampfactor = dampfactor
        self.maxite = maxite
        self.getsomeparam()

    def addDefect(self,param_defect={'type':'D','Et':0,'sigma_e':1e-12,'sigma_h':1e-12,'Nt':1e12}):
        '''
        Add a defect to the sample, each defect is a dictionary contains the parameters
        '''
        self.defect_list.append(param_defect)
        self.SolveEq()

    def PrintDefectList(self):
        '''
        Print the current defects list
        '''
        for i in range(len(self.defect_list)):
            print('Defect No.'+str(i)+':',self.defect_list[i])

    def DelDefect(self,index):
        '''
        Delete a defect from the defect list
        '''
        del self.defect_list[index]
        self.SolveEq()

    def getsomeparam(self):
        '''
        Calculate some basic parameters
        Note that the mobility is calculated for thermal equilibrium and only 
        considers the base doping, i.e., the defect induced doping is not considered
        '''
        ni = NI().update(temp=self.temp)
        self.ni = ni[0]
        ve, self.vh = Vel_th().update(temp=self.temp)
        self.ve = ve[0]
        if self.doptype == 'p':
            Na = Ion(temp=self.temp).update_dopant_ionisation(
                N_dop=self.Ndop, nxc=0, impurity='boron')
            self.Na_b = Na[0]
            self.Nd_b = 0.1
        elif self.doptype == 'n':
            Nd = Ion(temp=self.temp).update_dopant_ionisation(
                N_dop=self.Ndop, nxc=0, impurity='phosphorous')
            self.Nd_b = Nd[0]
            self.Na_b = 0.1
        self.miu_tot = mob(temp=self.temp, Na=self.Na_b, Nd=self.Nd_b).mobility_sum()
        self.miu_h = mob(temp=self.temp, Na=self.Na_b, Nd=self.Nd_b).hole_mobility()
        self.miu_e = mob(temp=self.temp, Na=self.Na_b, Nd=self.Nd_b).electron_mobility()


    def SolveEq(self):
        '''
        Solve the thermal equilibrum carrier concentrations and defects' occupancy
        '''
        self.getsomeparam()
        self.f0list = [] # List of occupancy
        for d in self.defect_list:
            self.f0list.append([1, 0, 0])
        diff = 100
        n00 = self.Nd_b
        p00 = self.Na_b
        ite_num = 0
        while diff > self.tol and ite_num<self.maxite:
            Ndtot1 = 0
            Natot1 = 0
            for d, f0 in zip(self.defect_list,self.f0list):
                if d['type'] == 'D':
                    Ndtot1 += f0[1] * d['Nt']
                elif d['type'] == 'A':
                    Natot1 += f0[0] * d['Nt']
                elif d['type'] == 'AD':
                    Ndtot1 += f0[2] * d['Nt']
                    Natot1 += f0[0] * d['Nt']
                elif d['type'] == 'DD':
                    Ndtot1 += 2 * f0[2] * d['Nt']
                    Ndtot1 += f0[1] * d['Nt']
                elif d['type'] == 'AA':
                    Natot1 += f0[1] * d['Nt']
                    Natot1 += 2 * f0[0] * d['Nt']
            Ndtot = self.Nd_b + Ndtot1
            Natot = self.Na_b + Natot1
            diffn0 = 0.5 * ((Ndtot - Natot) + np.sqrt((Ndtot - Natot)**2 + 4 * self.ni**2)) - n00
            diffp0 = 0.5 * ((Natot - Ndtot) + np.sqrt((Natot - Ndtot)**2 + 4 * self.ni**2)) - p00
            if Ndtot >= Natot:
                self.n0 = 0.5 * ((Ndtot - Natot) +
                             np.sqrt((Ndtot - Natot)**2 + 4 * self.ni**2)) - (1-self.dampfactor)*diffn0
                self.p0 = self.ni**2 / self.n0
            elif Ndtot < Natot:
                self.p0 = 0.5 * ((Natot - Ndtot) +
                             np.sqrt((Natot - Ndtot)**2 + 4 * self.ni**2)) - (1-self.dampfactor)*diffp0
                self.n0 = self.ni**2 / self.p0
            n00 = self.n0
            p00 = self.p0
            fnlist = []
            for x,f0 in zip(self.defect_list,self.f0list):
                if x['type'] == 'DD' or x['type'] == 'AA' or x['type'] == 'AD':
                    FFF = 1 + self.n0 / self.ni / \
                        np.exp(x['Et'][0] / kb / self.temp) + self.ni*\
                        np.exp(x['Et'][1] / kb / self.temp) / self.n0
                    damp = self.n0 / self.ni / np.exp(x['Et'][0] / kb / self.temp) / FFF - f0[0]
                    fnlist.append([self.n0 / self.ni / np.exp(x['Et'][0] / kb / self.temp) / FFF-0.01*damp,
                                   1 / FFF, self.ni / self.n0 * np.exp(x['Et'][1] / kb / self.temp) / FFF+0.01*damp])
                else:
                    damp = 1 / (1 + np.exp(x['Et'] / kb / self.temp) * self.ni / self.n0) - f0[0]
                    fnlist.append([1 / (1 + np.exp(x['Et'] / kb / self.temp) * self.ni / self.n0)-0.*damp,\
                                   1 / (1 + self.n0/np.exp(x['Et'] / kb / self.temp) / self.ni) + 0.*damp,0])
            diff = 0
            for fn, f, d in zip(fnlist, self.f0list, self.defect_list):
                if d['type'] == 'D' or d['type'] == 'A':
                    diff += abs(fn[0] - f[0])
                else:
                    diff += abs(fn[0] - f[0]) + abs(fn[-1] - f[-1])
            self.f0list = np.asarray(fnlist)
            ite_num += 1
        if diff > self.tol:
            print("May not convergy when solving thermal equilibrium")
        if (self.doptype =='n' and self.n0<self.p0) or (self.doptype =='p' and self.p0<self.n0):
            print("The sample is compensated by defect in thermal equilibrium")


    def calculatent(self,flist):
        '''
        A easy function to calculate the captured eletron cocentration in each
        defect
        '''
        ntlist = []
        for f, d in zip(flist, self.defect_list):
            if d['type'] == 'AD' or d['type'] == 'DD' or d['type'] == 'AA':
                # print(f)
                ntlist.append((2 * f[0] + f[1]) * d['Nt'])
            else:
                ntlist.append(f[0] * d['Nt'])
        return ntlist


    def calculatentDefectChargelist(self,flist):
        '''
        A easy function to calculate the total charge concentration on each
        defect
        '''
        dchargelist = []
        for d, fnn in zip(self.defect_list, flist):
            dcharge = 0
            if d['type'] == 'D':
                dcharge += d['Nt'] * fnn[1]
            elif d['type'] == 'A':
                dcharge += -d['Nt'] * fnn[0]
            elif d['type'] == 'DA':
                dcharge += d['Nt'] * (fnn[2] - fnn[0])
            elif d['type'] == 'DD':
                dcharge += -d['Nt'] * (2 * fnn[0] + fnn[1])
            elif d['type'] == 'AA':
                dcharge += d['Nt'] * (fnn[1] + 2 * fnn[2])
            dchargelist.append(dcharge)
        return dchargelist


    def calculateChargeNeutrality(self, nlist, plist, flist, **kwarg):
        '''
        Calculate the net charge in the sample. Should be a list of values close
        to zero if charge neutrality is fullfilled
        '''
        dn = nlist-self.n0
        dp = plist-self.p0
        nt0 = np.sum(self.calculatent(self.f0list))
        dnt = []
        for fflist in flist:
            dnt.append(np.sum(self.calculatent(fflist)) - nt0)
        dnt = np.asarray(dnt)
        chargelist = dp-dn-dnt
        return chargelist


    def SolveSS(self, nxc = [1e15]):
        '''
        Solve the steady state carrier concentrations and defects' occupancy
        The input is a list of excess minority carrier concentration, here the
        minority carrier is defined based on the background doping and does not 
        consider the doping induced by defect
        '''
        if isinstance(nxc,int) or isinstance(nxc,float):
            nxc = [nxc]
        self.SolveEq()
        nt0sum = sum(self.calculatent(self.f0list))
        nlist = []
        plist = []
        flist = []
        for i in range(len(nxc)):
            dx = nxc[i]
            diff = 100
            n = self.n0 + dx
            p = self.p0 + dx
            ite_num = 0
            while diff > self.tol and ite_num < self.maxite:
                fnnlist = []
                for x in self.defect_list:
                    if x['type'] == 'AD' or x['type'] == 'DD' or x['type'] == 'AA':
                        FFF = 1 + ((x['sigma_e'][0] * self.ve * n + x['sigma_h'][0] * \
                        self.vh * self.ni * np.exp(-x['Et'][0] / kb / self.temp)) / \
                        (x['sigma_h'][0] * self.vh * p + x['sigma_e'][0] * self.ve * self.ni * \
                        np.exp(x['Et'][0] / kb / self.temp))) + \
                        ((x['sigma_h'][1] * self.vh * p + x['sigma_e'][1] * self.ve * self.ni *\
                         np.exp(x['Et'][1] / kb / self.temp)) /\
                        (x['sigma_e'][1] * self.ve * n + x['sigma_h'][1] * self.vh * self.ni *\
                         np.exp(-x['Et'][1] / kb / self.temp)))
                        fnnlist.append([1 / FFF * ((x['sigma_e'][0] * self.ve * n + \
                        x['sigma_h'][0] * self.vh * self.ni * np.exp(-x['Et'][0] / kb / self.temp)) / \
                        (x['sigma_h'][0] * self.vh * p + x['sigma_e'][0] * self.ve * self.ni * \
                        np.exp(x['Et'][0] / kb / self.temp))), 1 / FFF, 1 / FFF *\
                         ((x['sigma_h'][1] * self.vh * p + x['sigma_e'][1] * self.ve * self.ni * \
                         np.exp(x['Et'][1] / kb / self.temp)) / (x['sigma_e'][1] * \
                         self.ve * n + x['sigma_h'][1] * self.vh * self.ni * \
                         np.exp(-x['Et'][1] / kb / self.temp)))])
                    else:
                        fnnlist.append([((x['sigma_e'] * self.ve * n + x['sigma_h'] \
                        * self.vh * self.ni * np.exp(-x['Et'] / kb / self.temp)) /
                        (x['sigma_e'] * self.ve * (n + self.ni * np.exp(x['Et'] / kb / self.temp)) +\
                        x['sigma_h'] * self.vh * (p + self.ni * np.exp(-x['Et'] / kb / self.temp)))),\
                        1-((x['sigma_e'] * self.ve * n + x['sigma_h'] \
                        * self.vh * self.ni * np.exp(-x['Et'] / kb / self.temp)) /
                        (x['sigma_e'] * self.ve * (n + self.ni * np.exp(x['Et'] / kb / self.temp)) +\
                        x['sigma_h'] * self.vh * (p + self.ni * np.exp(-x['Et'] / kb / self.temp)))),0])
                ntsum = sum(self.calculatent(fnnlist))
                if self.doptype == 'p':
                    pp = (self.p0 + dx + (ntsum - nt0sum))
                    diff = np.sum(abs(pp - p) / p)
                    p = pp
                elif self.doptype == 'n':
                    nn = (self.n0 + dx - (ntsum - nt0sum))
                    diff = np.sum(abs(nn - n) / n)
                    n = nn
                ite_num +=1
            if diff > self.tol:
                print("May not convergy when solving thermal equilibrium")
            nlist.append(n)
            plist.append(p)
            flist.append(fnnlist)
        flist=np.asarray(flist)
        return nlist, plist, flist


    def SolveTransient(self, t, n_initial, p_initial, flist_initial, fillG=1e20, opt='decay',**kwarg):
        '''
        Solve the transiente carrier concentrations and defects' occupancy
        The input is the initial carrier concentrations and defect occupancy
        '''
        self.SolveEq()
        initalcharge = self.calculateChargeNeutrality(nlist=[n_initial], plist=[p_initial], flist=[flist_initial])
        if initalcharge[0]>1000:
            warnings.warn('Inital charge = {:.2e} \nCheck the charge neutrality of the inital state'.format(initalcharge[0]),UserWarning)

        y0 = [n_initial, p_initial]
        for x,f in zip(self.defect_list,flist_initial):
            if x['type'] == 'AD' or x['type'] == 'DD' or x['type'] == 'AA':
                y0.append(x['Nt']*f[0])
                y0.append(x['Nt']*f[2])
            else:
                y0.append(x['Nt']*f[0])

        if opt=='decay':
            Gen=0
            gen = 0*t
        elif opt=='fill':
            Gen=fillG
            gen = 0*t+fillG
        # ODE
        def trapode(y, t, Gen):
            n = y[0]
            p = y[1]
            dndt=0
            dpdt=0
            currentidx=2
            dydt=[Gen,Gen]
            for x in self.defect_list:
                if x['type'] == 'AD' or x['type'] == 'DD' or x['type'] == 'AA':
                    dndt += x['sigma_e'][0] * self.ve *(self.ni*np.exp(x['Et'][0] / kb / self.temp)*\
                            y[currentidx]-n*(x['Nt']-y[currentidx]-y[currentidx+1]))+\
                            x['sigma_e'][1] * self.ve *((x['Nt']-y[currentidx]-y[currentidx+1])*\
                            self.ni*np.exp(x['Et'][1] / kb / self.temp)-n*y[currentidx+1])
                    dpdt += x['sigma_h'][0] * self.vh *(self.ni * np.exp(-x['Et'][1] / kb / self.temp)*\
                            (x['Nt']-y[currentidx]-y[currentidx+1])-p*y[currentidx])+\
                            x['sigma_h'][1] * self.vh *(y[currentidx+1]*self.ni * np.exp(-x['Et'][1] / kb / self.temp)-\
                            p*(x['Nt']-y[currentidx]-y[currentidx+1]))
                    dydt.append((x['sigma_h'][0] * self.vh *(self.ni * np.exp(-x['Et'][1] / kb / self.temp)*\
                                (x['Nt']-y[currentidx]-y[currentidx+1])-p*y[currentidx]))-(x['sigma_e'][0] *\
                                self.ve *(self.ni*np.exp(x['Et'][0] / kb / self.temp)*\
                                y[currentidx]-n*(x['Nt']-y[currentidx]-y[currentidx+1]))))
                    dydt.append((x['sigma_e'][1] * self.ve *((x['Nt']-y[currentidx]-y[currentidx+1])*\
                                self.ni*np.exp(x['Et'][1] / kb / self.temp)-n*y[currentidx+1]))-(x['sigma_h'][1] * \
                                self.vh *(y[currentidx+1]*self.ni * np.exp(-x['Et'][1] / kb / self.temp)-\
                                p*(x['Nt']-y[currentidx]-y[currentidx+1]))))
                    currentidx +=2
                else:
                    dndt += x['sigma_e'] * self.ve *(self.ni*np.exp(x['Et'] / kb / self.temp)*\
                            y[currentidx]-n*(x['Nt']-y[currentidx]))
                    dpdt += x['sigma_h'] * self.vh *(self.ni * np.exp(-x['Et'] / kb / self.temp)*\
                            (x['Nt']-y[currentidx])-p*y[currentidx])
                    dydt.append((x['sigma_h'] * self.vh *(self.ni * np.exp(-x['Et'] / kb / self.temp)*\
                            (x['Nt']-y[currentidx])-p*y[currentidx]))-(x['sigma_e'] * self.ve *(self.ni*np.exp(x['Et'] / kb / self.temp)*\
                                    y[currentidx]-n*(x['Nt']-y[currentidx]))))
                    currentidx +=1

            dydt[0] = Gen + dndt
            dydt[1] = Gen + dpdt
            return dydt

        sol = odeint(trapode, y0, t, args=(Gen,), rtol=self.tol, atol=self.tol)

        nlist = sol[:, 0]
        plist = sol[:, 1]
        nt = sol[:, 2:]
        flist=[]
        for i in range(len(t)):
            fnnlist=[]
            currentidx=2
            for x in self.defect_list:
                if x['type'] == 'AD' or x['type'] == 'DD' or x['type'] == 'AA':
                    fnnlist.append([sol[i,currentidx]/x['Nt'],1-sol[i,currentidx]/x['Nt']-sol[i,currentidx+1]/x['Nt'],sol[i,currentidx+1]/x['Nt']])
                    currentidx += 2
                else:
                    fnnlist.append([sol[i,currentidx]/x['Nt'],1-sol[i,currentidx]/x['Nt'],0])
                    currentidx += 1
            flist.append(fnnlist)
        flist=np.asarray(flist)
        return nlist, plist, flist, gen


    def calculateRate(self,nlist, plist, flist, **kward):
        '''
        Calculate the four rates at a defect: electron/hole capture 
        and electron/hole emission and the recombinaiton rate
        '''
        Gelist = []
        Relist = []
        Ghlist = []
        Rhlist = []
        Utotlist = []
        for n, p, fnnlist in zip(nlist, plist, flist):
            Ge = []
            Re = []
            Gh = []
            Rh = []
            Utot = 0
            for fnn, d in zip(fnnlist, self.defect_list):
                if d['type'] == 'AD' or d['type'] == 'DD' or d['type'] == 'AA':
                    Re.append([d['sigma_e'][0] * self.ve * n * d['Nt'] * (fnn[1]),
                               d['sigma_e'][1] * self.ve * n * d['Nt'] * (fnn[2])])
                    Ge.append([d['sigma_e'][0] * self.ve * self.ni * np.exp(d['Et'][0] / kb / self.temp) * d['Nt'] * fnn[0],
                               d['sigma_e'][1] * self.ve * self.ni * np.exp(d['Et'][1] / kb / self.temp) * d['Nt'] * fnn[1]])
                    Rh.append([d['sigma_h'][0] * self.vh * p * d['Nt'] * fnn[0],
                               d['sigma_h'][1] * self.vh * p * d['Nt'] * fnn[1]])
                    Gh.append([d['sigma_h'][0] * self.vh * self.ni * np.exp(-d['Et'][0] / kb / self.temp) * d['Nt'] * fnn[1],
                               d['sigma_h'][1] * self.vh * self.ni * np.exp(-d['Et'][1] / kb / self.temp) * d['Nt'] * fnn[2]])
                    Utot -= d['Nt']*(d['sigma_e'][0]*self.ve*(self.ni*np.exp(d['Et'][0]/kb/self.temp)*fnn[0]-n*fnn[1])+\
                            d['sigma_e'][1]*self.ve*(self.ni*np.exp(d['Et'][1]/kb/self.temp)*fnn[1]-n*fnn[2]))
                else:
                    Re.append(d['sigma_e'] * self.ve * n * d['Nt'] * fnn[1])
                    Ge.append(d['sigma_e'] * self.ve * self.ni *
                              np.exp(d['Et'] / kb / self.temp) * (d['Nt'] * fnn[0]))
                    Rh.append(d['sigma_h'] * self.vh * p * d['Nt'] * fnn[0])
                    Gh.append(d['sigma_h'] * self.vh * self.ni *
                              np.exp(-d['Et'] / kb / self.temp) * d['Nt'] * fnn[1])
                    Utot += d['sigma_e'] * self.ve * n * d['Nt'] * fnn[1] - d['sigma_e'] * self.ve * self.ni * np.exp(
                        d['Et'] / kb / self.temp) * (d['Nt'] * fnn[0])
            Gelist.append(Ge)
            Relist.append(Re)
            Ghlist.append(Gh)
            Rhlist.append(Rh)
            Utotlist.append(Utot)
        Gelist=np.asarray(Gelist)
        Relist=np.asarray(Relist)
        Ghlist=np.asarray(Ghlist)
        Rhlist=np.asarray(Rhlist)
        Utotlist=np.asarray(Utotlist)    
        return Gelist, Relist, Ghlist, Rhlist, Utotlist


    def calculateSSlifetime(self, nlist, plist, flist, **kward):
        '''
        Calculate the steady state lifetime
        '''
        Utotlist = self.calculateRate(nlist, plist, flist)[-1]
        condlist = []
        dminorlist = []
        tauminorlist = []
        dmajlist = []
        taumajorlist = []
        dapplist = []
        tauapplist = []
        for n, p, Utot in zip(nlist, plist, Utotlist):
            condlist.append(
                const.e * (self.miu_e * (n - self.n0) + self.miu_h * (p - self.p0)))
            dapplist.append(((n - self.n0) * self.miu_e + (p - self.p0) * self.miu_h) / self.miu_tot)
            tauapplist.append(
                (((n - self.n0) * self.miu_e + (p - self.p0) * self.miu_h) / self.miu_tot) / Utot)
            if self.doptype == 'n':
                dminorlist.append(p - self.p0)
                tauminorlist.append((p - self.p0) / Utot)
                dmajlist.append(n - self.n0)
                taumajorlist.append((n - self.n0) / Utot)
            elif self.doptype == 'p':
                dminorlist.append(n - self.n0)
                tauminorlist.append((n - self.n0) / Utot)
                dmajlist.append(p - self.p0)
                taumajorlist.append((p - self.p0) / Utot)
        return dminorlist, tauminorlist, dmajlist, taumajorlist, dapplist, tauapplist, condlist


    def calculateTranslifetime(self, nlist, plist, t, gen, **kward):
        '''
        Calculate the transient/filling lifetime
        '''
        condlist = const.e * (self.miu_e * (nlist - self.n0) + self.miu_h * (plist - self.p0))
        dapplist = (self.miu_e * (nlist - self.n0) + self.miu_h * (plist - self.p0))/self.miu_tot
        tauapplist= dapplist[1:]/(gen[1:]-np.diff(dapplist) / np.diff(t))
        if self.doptype == 'n':
            dminorlist = plist - self.p0
            tauminorlist= dminorlist[1:]/(gen[1:]-np.diff(plist) / np.diff(t))
            dmajlist = nlist - self.n0
            taumajorlist = dmajlist[1:]/(gen[1:]-np.diff(nlist) / np.diff(t))
        elif self.doptype == 'p':
            dminorlist = nlist - self.n0
            tauminorlist= dminorlist[1:]/(gen[1:]-np.diff(nlist) / np.diff(t))
            dmajlist = plist - self.p0
            taumajorlist = dmajlist[1:]/(gen[1:]-np.diff(plist) / np.diff(t))
        tauapplist = np.append([tauapplist[0]],tauapplist)
        tauminorlist = np.append([tauminorlist[0]],tauminorlist)
        taumajorlist = np.append([taumajorlist[0]],taumajorlist)
        return dminorlist, tauminorlist, dmajlist, taumajorlist, dapplist, tauapplist, condlist


    def singleSRH(self, nxc, defect, **kward):
        '''
        Calculate the analytic SRH lifetime for a single level defect
        '''
        alpha_e = defect['sigma_e'] * self.ve
        alpha_h = defect['sigma_h'] * self.vh
        n1 = self.ni * np.exp(defect['Et'] / kb / self.temp)
        p1 = self.ni * np.exp(-defect['Et'] / kb / self.temp)
        n = self.n0 + nxc
        p = self.p0 + nxc
        R = (n * p - self.ni**2) * alpha_e * alpha_h * \
            defect['Nt'] / ((n + n1) * alpha_e + (p + p1) * alpha_h)
        tau = nxc / R
        return tau

    def twolevelSRH(self, nxc, defect, **kwarg):
        '''
        Calculate the analytic Sah_Shockley lifetime for two-levels defect
        '''
        alpha_e1 = defect['sigma_e'][0] * self.ve
        alpha_h1 = defect['sigma_h'][0] * self.vh
        alpha_e2 = defect['sigma_e'][1] * self.ve
        alpha_h2 = defect['sigma_h'][1] * self.vh
        n1 = self.ni * np.exp(defect['Et'][0] / kb / self.temp)
        p1 = self.ni * np.exp(-defect['Et'][0] / kb / self.temp)
        n2 = self.ni * np.exp(defect['Et'][1] / kb / self.temp)
        p2 = self.ni * np.exp(-defect['Et'][1] / kb / self.temp)
        n = self.n0 + nxc
        p = self.p0 + nxc
        R = defect['Nt'] * (n * p - self.ni**2) / (1 + ((alpha_e2 * n2 + \
            alpha_h2 * p) / (alpha_e2 * n + alpha_h2 * p2)) + ((alpha_e1 * n \
            + alpha_h1 * p1) / (alpha_e1 *n1 + alpha_h1 * p))) * \
            ((alpha_e2 * alpha_h2 / (alpha_e2 * n + alpha_h2 * p2)) + \
            (alpha_e1 * alpha_h1 / (alpha_e1 * n1 + alpha_h1 * p)))
        tau = nxc / R
        return tau
