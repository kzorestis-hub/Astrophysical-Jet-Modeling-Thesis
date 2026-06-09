import os

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 1. HELPER FUNCTION: CALCULATE SLOPE
# ==============================================================================
Z_0 = 3.086e21
z_values = np.geomspace(Z_0,300.0*Z_0, 1000)

def calculate_slope(x_data, y_data):
    """
    Calculates the slope of the line in log-log space.
    Assumes x_data and y_data are numpy arrays or lists.
    """
    # 1. Take the log10 of the data
    # (Use a small number like 1e-99 to avoid log(0) errors if needed)
    log_x = np.log10(np.array(x_data) + 1e-99)
    log_y = np.log10(np.array(y_data) + 1e-99)

    # 2. Filter out any infinite/NaN values that might ruin the fit
    # (This happens if you have zeros in your data)
    valid_mask = np.isfinite(log_x) & np.isfinite(log_y)
    log_x = log_x[valid_mask]
    log_y = log_y[valid_mask]

    if len(log_x) < 2:
        return 0.0 # Not enough data points

    # 3. Perform a linear fit (degree=1)
    # polyfit returns [slope, intercept]
    slope, intercept = np.polyfit(log_x, log_y, 1)

    return slope



# ==============================================================================
# 2. THE PHYSICS ENGINE (Kinematic/Passive)
# ==============================================================================

class RelativisticJetCalculator:
    def __init__(self, pressure_law_func, p_total_0, rho0, v0, r0, b0, gamma_ad):
        self.pressure_law_func = pressure_law_func
        self.P_total_0 = p_total_0
        self.rho0 = rho0
        self.v0 = v0
        self.r0 = r0
        self.B_0 = b0
        self.c = 2.9979e10

        # Physics setup
        self.gamma_adiabatic = gamma_ad
        self.p_exponent = 1.0 / self.gamma_adiabatic
        self.h_factor = self.gamma_adiabatic / (self.gamma_adiabatic - 1.0)

        self.gamma0 = 1.0 / np.sqrt(1.0 - (self.v0 / self.c)**2)
        self.A0 = np.pi * self.r0**2

        # Kinematic Assumption
        self.P_th_0 = self.P_total_0

        # Constants
        self.B_const = self.rho0 / (self.P_th_0 ** self.p_exponent)
        self.h0_hydro = 1.0 + (self.h_factor * self.P_th_0) / (self.rho0 * self.c**2)

        self.C1_const = self.h0_hydro * self.gamma0
        self.C2_const = self.A0 * self.gamma0 * self.v0 * self.rho0
        self.C3_B_model = self.B_0 * np.sqrt(self.v0 * self.gamma0 / self.rho0)
        
        # Constant total luminosity: L_total = h * gamma^2 * A * v * rho * c^2
        self.L_total_const = self.h0_hydro * (self.gamma0**2) * self.A0 * self.v0 * self.rho0 * (self.c**2)



        # STATE TRACKING FOR CONICAL LOCK 
        self.conical_locked = False
        self.z_prev = None
        self.r_prev = None
        
        # Lock values to freeze when a=1 is reached
        self.z_lock = None
        self.r_lock = None
        
        # State tracking for previous gamma (needed for conical lock binary search)
        self.gamma_prev = self.gamma0
        self.opening_angle = None  # Will be computed once conical lock engages

    def get_properties(self, z):

        # Test results
        P_z_test = self.pressure_law_func(z)
        if P_z_test < 0.0: return None

        rho_test = self.B_const * (P_z_test ** self.p_exponent)
        h_test = 1.0 + (self.h_factor * P_z_test) / (rho_test * self.c**2)
        gamma_test = self.C1_const / h_test
        v_test = self.c * np.sqrt(1.0 - 1.0 / gamma_test**2)
        denom_test = rho_test * gamma_test * v_test
        area_test = self.C2_const / denom_test
        radius_test = np.sqrt(area_test / np.pi)
        cs_test = np.sqrt(self.gamma_adiabatic * P_z_test / (rho_test * h_test))
        gamma_cs_test = np.sqrt(1 / (1 - (cs_test / self.c)**2))
        theta_M = gamma_cs_test * cs_test / (gamma_test * v_test) # mach cone angle = arcsin (theta_M) = theta_M , for theta_M small
             # 2. CHECK THE LIMIT (a >= 1)


    
        
        if not self.conical_locked and self.z_prev is not None and self.r_prev is not None:
            a_local = np.log(radius_test / self.r_prev) / np.log(z / self.z_prev)
            
            if a_local >= 1.0:
                self.conical_locked = True
                self.z_lock = self.z_prev
                self.r_lock = self.r_prev


        '''
        
        if not self.conical_locked and self.z_prev is not None and self.r_prev is not None:
            dr_dz = (radius_test - self.r_prev) / (z - self.z_prev)

            if dr_dz >= theta_M:   # jet opening angle assumed to be small so tan(thetaj) approx thetaj 
                self.conical_locked = True
                self.z_lock = self.z_prev
                self.r_lock = self.r_prev
                print(f"theta_M = {theta_M}")
        '''

        if self.conical_locked:
            radius = (self.r_lock / self.z_lock) * z
            area = np.pi * radius**2

            # Use a binary search to find the exact gamma that conserves energy
            gamma_low = self.gamma_prev  # only goes up
            gamma_high = self.C1_const  # The absolute maximum possible gamma

            for _ in range(40):
                gamma_mid = 0.5 * (gamma_low + gamma_high)
                v_mid = self.c * np.sqrt(1.0 - 1.0 / gamma_mid**2)
                rho_mid = self.C2_const / (area * gamma_mid * v_mid)
                P_mid = (rho_mid / self.B_const)**(1.0 / self.p_exponent)
                h_mid = 1.0 + (self.h_factor * P_mid) / (rho_mid * self.c**2)

                if h_mid * gamma_mid > self.C1_const:
                    gamma_high = gamma_mid
                else:
                    gamma_low = gamma_mid

            

            gamma = 0.5 * (gamma_low + gamma_high)
            v = self.c * np.sqrt(1.0 - 1.0 / gamma**2)
            rho = self.C2_const / (area * gamma * v)
            P_z = (rho / self.B_const)**(1.0 / self.p_exponent)
            h = 1.0 + (self.h_factor * P_z) / (rho * self.c**2)
        else:
            P_z = P_z_test
            rho = rho_test
            h = h_test
            gamma = gamma_test
            v = v_test
            radius = radius_test
            area = area_test


        if abs(h * gamma - self.C1_const) / self.C1_const > 0.01:
            print(f"Warning: Energy not conserved in binary search at z={z}")
        
        self.z_prev = z
        self.r_prev = radius
        self.gamma_prev = gamma  # Store gamma for next iteration's conical lock binary search

        # B-Field
        B_prime = self.C3_B_model * np.sqrt(rho / (v * gamma))

        # 4. Energy Fluxes
        # L_total = L_restmass + L_kin + L_th = constant
        # where L_restmass = M_dot * c^2 = rho * A * vp * gamma * c^2
        l_restmass = rho * area * v * gamma * (self.c**2.0)
        l_kin = (gamma - 1.0) * self.C2_const * self.c**2.0
        l_th = (h - 1.0) * self.C2_const * self.c**2.0 * gamma
        l_mag = (B_prime**2.0 / (4.0 * np.pi)) * v * (gamma**2.0) * area
        l_total = l_kin + l_th 

        # Mass rate (should be constant by mass conservation)
        # M_dot = rho * A * v * gamma = C2_const
        M_dot = gamma * v * area * rho

        # Energy rate without magnetic field (should be constant by energy conservation)
        # L_no_mag = L_kin + L_th
        L_no_mag = l_kin + l_th

        # Local Sound speed
        c_s = np.sqrt(self.gamma_adiabatic * P_z / (rho * h))
        # 'Lab' Sound speed
        c_s_lab = c_s / gamma
        q_e = 4.80320425e-10  # Statcoulombs (esu)

        # Radiation Spectrum per Bin
        # Electron internal energy fraction per bin, flow frame
        dz_array = np.gradient(z_values)
        d = 30 * 3.086e24  # Distance in cm
        area_constant = 4.0 * np.pi * d**2.0
        # Find the index of current z using binary search (much faster than argmin)
        z_index = np.searchsorted(z_values, z)
        z_index = min(z_index, len(z_values) - 1)  # Clamp to valid range
        dz = dz_array[z_index]
        bita = v / self.c
        eta = 0.1
        U_int = 3.0 * P_z
        U_e_cell = eta * U_int * dz * np.pi * radius**2 *gamma  # Internal energy in the cell available for electrons, flow frame. dz was comoving so we add gamma factor to get the lab frame volume element, which is what matters for the energy content of the cell.
        m_e = 9.10938356e-28
        sigma_T = 6.6524587158e-25
        # Calculate temperature in the cell
        mu = 0.5  # Mean molecular weight (ionized plasma)
        m_p = 1.673e-24  # Proton mass in grams
        k_B = 1.381e-16  # Boltzmann constant in erg/K
        T_cell = (U_int * mu * m_p) / (3* rho * k_B)  # Temperature in Kelvin
        # Calculate thermal Lorentz factor from temperature
        gamma_T = 1.0 + (k_B * T_cell) / (m_e * self.c**2)
        # Define the range of electron energies (E_e_min to E_e_max) based on physical considerations
        E_e_min = (gamma_T) * m_e * self.c**2.0
        E_e_max = 1e6 * m_e * self.c**2.0
        # Define the range of frequencies corresponding to E_e_min and E_e_max
        n_c_min = 3.0 / (4.0 * np.pi) * q_e * B_prime * E_e_min**2.0 / (m_e**3.0 * self.c**5.0)
        n_c_max = 3.0 / (4.0 * np.pi) * q_e * B_prime * E_e_max**2.0 / (m_e**3.0 * self.c**5.0)
        D_n = (np.log(n_c_max) - np.log(n_c_min)) / 499  # freq step in log space
        # now in the observers frame, doppler
        # defining range of frequencies
        
        N_o = U_e_cell / (np.log(E_e_max / E_e_min))
        E_values = np.geomspace(E_e_min, E_e_max, 500)
        n_c_list_thita1 = []
        n_c_list_thita2 = []
        nLn_ff_list = []
        nLn_app_list_thita1 = []
        nLn_app_list_thita2 = []
        F_obs_1 = []
        F_obs_2 = []
        
        for i in range(len(E_values)):
            E = E_values[i]
            gamma_e = E / (m_e * self.c**2.0)
            bita_e = np.sqrt(1.0 - 1.0 / gamma_e**2.0)
            DEradDt = (4.0 / 3.0) * sigma_T * bita_e**2.0 * self.c * (B_prime**2.0 / (8.0 * np.pi)) * (E / (m_e * self.c**2.0))**2.0
            N_e = N_o * (E**-2.0)
            nLn = 0.5 * E * N_e * DEradDt  # flow frame
            n_c = 3.0 / (4.0 * np.pi) * q_e * B_prime * E**2.0 / (m_e**3.0 * self.c**5.0)
            nLn_ff_list.append(nLn)  # remember step is logarithmic
            
            

            for j in range(2):
                thita = 10.0 * np.pi / 180.0 + 35.0 * j * np.pi / 180.0  # 10 degrees in radians
                delta = 1.0 / (gamma * (1.0 - bita * np.cos(thita)))  # Assuming observer is along the jet axis (theta=0)
                nLn_obs = nLn * delta**3.0 / gamma  # Doppler boosting for luminosity
                n_c_obs = n_c * delta  # Doppler boosting for frequency

                if j == 0:
                    n_c_list_thita1.append(n_c_obs)
                    nLn_app_list_thita1.append(nLn_obs)
                else:
                    n_c_list_thita2.append(n_c_obs)
                    nLn_app_list_thita2.append(nLn_obs)


        # Calculate Bolometric Luminosity ONCE after energy loop completes
        # Use proper trapezoidal rule: (f[0] + f[-1])/2 + sum(f[1:-1])
        if len(nLn_app_list_thita1) > 1:
            Sum_L_1 = (nLn_app_list_thita1[0] + nLn_app_list_thita1[-1]) / 2.0 + np.sum(nLn_app_list_thita1[1:-1])
            L_bol_app_1 = Sum_L_1 * D_n
        else:
            L_bol_app_1 = 0.0

        if len(nLn_app_list_thita2) > 1:
            Sum_L_2 = (nLn_app_list_thita2[0] + nLn_app_list_thita2[-1]) / 2.0 + np.sum(nLn_app_list_thita2[1:-1])
            L_bol_app_2 = Sum_L_2 * D_n
        else:
            L_bol_app_2 = 0.0

        # Calculate comoving frame bolometric luminosity
        if len(nLn_ff_list) > 1:
            Sum_L = (nLn_ff_list[0] + nLn_ff_list[-1]) / 2.0 + np.sum(nLn_ff_list[1:-1])
            total_lum_comoving = Sum_L * D_n
        else:
            total_lum_comoving = 0.0
        
        L_lab = total_lum_comoving  # Total luminosity in Lab Frame, according to the integral I solved!

        # Turning Luminosity into Flux at Earth (assuming d=30 Mpc)
        Ln_array_1 = np.array(nLn_app_list_thita1)
        nu_array_1 = np.array(n_c_list_thita1)
        Ln_array_2 = np.array(nLn_app_list_thita2)
        nu_array_2 = np.array(n_c_list_thita2)

        # --- Standard Spectral Flux Density (F_nu) in mJy ---
        F_obs_1 = Ln_array_1 / (nu_array_1 * area_constant) * 10**26  # Spectral flux density F_nu (mJy)
        F_obs_2 = Ln_array_2 / (nu_array_2 * area_constant) * 10**26  # Spectral flux density F_nu (mJy)
        F_obs_1_z = F_obs_1 * z / dz                                  # z * dF_nu/dz (mJy)
        F_obs_2_z = F_obs_2 * z / dz                                  # z * dF_nu/dz (mJy)

        # --- CORRECTED: Bolometric Flux (F_bol) as z * dF/dz in Absolute Physical Units ---
        # Removed 10**26 to keep in erg/s/cm^2 and multiplied by (z / dz) for log-distance scaling
        F_bol_obs_1 = (L_bol_app_1 / area_constant) * (z / dz)  # Total observer flux slice (erg/s/cm^2)
        F_bol_obs_2 = (L_bol_app_2 / area_constant) * (z / dz)  # Total observer flux slice (erg/s/cm^2)

        # --- CORRECTED: Spectral Energy Flux (nu * F_nu) as z * dF/dz in Absolute Physical Units ---
        # Removed 10**26 because nu*F_nu is measured in erg/s/cm^2, not mJy
        F_obs_1_nu = Ln_array_1 / area_constant                  # Intrinsic cell nu * F_nu (erg/s/cm^2)
        F_obs_2_nu = Ln_array_2 / area_constant                  # Intrinsic cell nu * F_nu (erg/s/cm^2)
        
        # Slices scale cleanly as z * d(nu * F_nu)/dz
        F_obs_1_z_nu = F_obs_1_nu * z / dz                       # Differential energy flux per log-interval
        F_obs_2_z_nu = F_obs_2_nu * z / dz                       # Differential energy flux per log-interval

        L_lab_scaled = L_lab * z / dz
        L_bol_app_1_scaled = L_bol_app_1 * z / dz
        L_bol_app_2_scaled = L_bol_app_2 * z / dz


        return {'z': z, 'gamma': gamma, 'radius': radius, 'B_prime': B_prime, 'n_c_list_thita1': n_c_list_thita1, 'nLn_app_list_thita1': nLn_app_list_thita1,
                'n_c_list_thita2': n_c_list_thita2, 'nLn_app_list_thita2': nLn_app_list_thita2,
                'P_z': P_z, 'l_kin': l_kin, 'l_th': l_th, 'l_mag': l_mag, 'L_lab_scaled': L_lab_scaled, 'L_bol_app_1_scaled': L_bol_app_1_scaled, 'L_bol_app_2_scaled': L_bol_app_2_scaled,
                'l_total': l_total, 'M_dot': M_dot, 'L_no_mag': L_no_mag, 'c_s': c_s, 'c_s_lab': c_s_lab, 'dz_array': dz_array, 'L_lab': L_lab, 'L_bol_app_2': L_bol_app_2, 'L_bol_app_1': L_bol_app_1, 'F_obs_1': F_obs_1, 'F_obs_2': F_obs_2, 'F_obs_1_z': F_obs_1_z, 'F_obs_2_z': F_obs_2_z, 'F_bol_obs_1': F_bol_obs_1, 'F_bol_obs_2': F_bol_obs_2, 'gamma_T': gamma_T, 'T_cell': T_cell, 'F_obs_1_z_nu': F_obs_1_z_nu, 'F_obs_2_z_nu': F_obs_2_z_nu}
# ==============================================================================
# 3. DEFINE SIMULATION PARAMETERS
# ==============================================================================

c = 2.9979e10
V_0 = c / np.sqrt(3.0) 
gamma_0 = 1.0 / np.sqrt(1.0 - (V_0/c)**2)  
h_gamma = 10.0
h_0 = h_gamma / gamma_0  # Will be ~8.165
R_0 = 3.086e20  # 100 parsecs in cm
M_dot = 10**44 / ((h_gamma-1)*c**2)  # Mass rate corresponding to a kinetic luminosity of 10^44 erg/s at the base (10^44 is without the rest mass)
RHO_0 = M_dot / (V_0*gamma_0*np.pi*R_0**2)
P_0 = (h_0 - 1.0) * (RHO_0 * c**2) / 4.0
B_0 =   np.sqrt(0.1*8.0 * np.pi  * P_0)  # 1% of the thermal pressure in Gauss
Z_0 = 3.086e21  # 1 kiloparsec in centimeters
Z_scale = Z_0

models = [
    {"name": "Model A:",      "z_scale":Z_0, "exponent": 3,"color": "#e74c3c", "ls": ":"},
    {"name": "Model B:", "z_scale":Z_0, "exponent": 2, "color": "#2ecc71", "ls": "-"},
    {"name": "Model C:",   "z_scale":Z_0, "exponent": 1.5, "color": "#3498db", "ls": "--"},
]

# ==============================================================================
# 4. RUN SIMULATIONS
# ==============================================================================

results = {m['name']: {'z': [], 'gamma': [], 'r': [], 'B': [], 'P_z': [], 'l_kin': [], 'l_th': [], 'l_mag': [], 'l_total': [], 'M_dot': [], 'L_no_mag': [], 'c_s': [], 'c_s_lab': [], 'n_c_list_thita1': [], 'nLn_app_list_thita1': [], 'n_c_list_thita2': [], 'nLn_app_list_thita2': [], 'L_lab': [], 'L_bol_app_1': [], 'L_bol_app_2': [], 'F_obs_1': [], 'F_obs_2': [], 'F_obs_1_z': [], 'F_obs_2_z': [], 'F_bol_obs_1': [], 'F_bol_obs_2': [], 'gamma_T': [], 'T_cell': [], 'F_obs_1_z_nu': [], 'F_obs_2_z_nu': [], 'L_bol_app_1_scaled': [], 'L_bol_app_2_scaled': [], 'L_lab_scaled': []} for m in models}

for model in models:
    print(f"Running {model['name']}...")

    def pressure_law(z):
        z_safe = z + 1e-99
        return 0.0 + (P_0 - 0.0) * (z_safe / model['z_scale'])**(-model['exponent'])

    engine = RelativisticJetCalculator(
        pressure_law_func=pressure_law,
        p_total_0=P_0, rho0=RHO_0, v0=V_0, r0=R_0, b0=B_0, gamma_ad=4.0/3.0
    )

    for z in z_values:
        data = engine.get_properties(z)
        if data:
            results[model['name']]['z'].append(z)
            results[model['name']]['gamma'].append(data['gamma'])
            results[model['name']]['r'].append(data['radius'])
            results[model['name']]['B'].append(data['B_prime'])
            results[model['name']]['P_z'].append(data['P_z'])
            results[model['name']]['l_kin'].append(data['l_kin'])
            results[model['name']]['l_th'].append(data['l_th'])
            results[model['name']]['l_mag'].append(data['l_mag'])
            # Optional totals returned by the engine
            if 'l_total' in data:
                results[model['name']]['l_total'].append(data['l_total'])
            if 'M_dot' in data:
                results[model['name']]['M_dot'].append(data['M_dot'])
            if 'L_no_mag' in data:
                results[model['name']]['L_no_mag'].append(data['L_no_mag'])
            if 'c_s' in data:
                results[model['name']]['c_s'].append(data['c_s'])
            if 'c_s_lab' in data:
                results[model['name']]['c_s_lab'].append(data['c_s_lab'])
            if 'n_c_list_thita1' in data and 'nLn_app_list_thita1' in data:
                results[model['name']]['n_c_list_thita1'].append(data['n_c_list_thita1'])
                results[model['name']]['nLn_app_list_thita1'].append(data['nLn_app_list_thita1'])
            if 'n_c_list_thita2' in data and 'nLn_app_list_thita2' in data:
                results[model['name']]['n_c_list_thita2'].append(data['n_c_list_thita2'])
                results[model['name']]['nLn_app_list_thita2'].append(data['nLn_app_list_thita2'])
            if 'L_lab' in data:
                results[model['name']]['L_lab'].append(data['L_lab'])
            if 'L_bol_app_1' in data:
                results[model['name']]['L_bol_app_1'].append(data['L_bol_app_1'])
            if 'L_bol_app_2' in data:
                results[model['name']]['L_bol_app_2'].append(data['L_bol_app_2'])
            if 'F_obs_1' in data:
                results[model['name']]['F_obs_1'].append(data['F_obs_1'])
            if 'F_obs_2' in data:
                results[model['name']]['F_obs_2'].append(data['F_obs_2'])
            if 'F_obs_1_z' in data:
                results[model['name']]['F_obs_1_z'].append(data['F_obs_1_z'])
            if 'F_obs_2_z' in data:
                results[model['name']]['F_obs_2_z'].append(data['F_obs_2_z'])
            if 'F_bol_obs_1' in data:
                results[model['name']]['F_bol_obs_1'].append(data['F_bol_obs_1'])
            if 'F_bol_obs_2' in data:
                results[model['name']]['F_bol_obs_2'].append(data['F_bol_obs_2'])
            if 'F_obs_1_nu' in data:
                results[model['name']]['F_obs_1_nu'].append(data['F_obs_1_nu'])
            if 'F_obs_1_z_nu' in data:
                results[model['name']]['F_obs_1_z_nu'].append(data['F_obs_1_z_nu'])
            if 'F_obs_2_nu' in data:
                results[model['name']]['F_obs_2_nu'].append(data['F_obs_2_nu'])
            if 'F_obs_2_z_nu' in data:
                results[model['name']]['F_obs_2_z_nu'].append(data['F_obs_2_z_nu'])
            # Debug: print first 5 radii for Model B
            if model['name'] == "Model A:" and len(results[model['name']]['r']) <= 5:
                print(f"Model A: z={z:.2e} cm, radius={data['radius']:.2e} cm")
            if 'gamma_T' in data and 'T_cell' in data :
                results[model['name']]['gamma_T'].append(data['gamma_T'])
                results[model['name']]['T_cell'].append(data['T_cell'])
            if 'L_lab_scaled' in data:
                results[model['name']]['L_lab_scaled'].append(data['L_lab_scaled'])
            if 'L_bol_app_1_scaled' in data:
                results[model['name']]['L_bol_app_1_scaled'].append(data['L_bol_app_1_scaled'])
            if 'L_bol_app_2_scaled' in data:
                results[model['name']]['L_bol_app_2_scaled'].append(data['L_bol_app_2_scaled'])

# ==============================================================================
# 5. PLOTTING (Bulletproof, Auto-Save, & Professor Updates)
# ==============================================================================
import os

# Define the output directory on the Desktop
desktop_path = os.path.expanduser("~/Desktop")
output_dir = os.path.join(desktop_path, "THESIS RESUTLS")
os.makedirs(output_dir, exist_ok=True) 

print(f"\nGenerating Plots and saving to: {output_dir}")

# Helper dictionary for P_ext slope (b values)
b_values = {'Model A': 3.0, 'Model B': 2.0, 'Model C': 1.5, 
            'Model A:': 3.0, 'Model B:': 2.0, 'Model C:': 1.5}

# ------------------------------------------------------------------------------
# PAIR 1: Acceleration (Gamma) and Radius (r)
# ------------------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
plt.subplots_adjust(wspace=0.25)

# --- Subplot 1a: Acceleration Profile (Gamma vs Z) ---
ax = axes1[0]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    gamma_data = np.array(results[name]['gamma'])
    
    if len(z_data) > 2:
        # DYNAMIC MASK: Only calculate the slope during the active acceleration phase!
        # z > 1.2 ignores the injection boundary, gamma_data < 8.0 ignores the saturation plateau
        mask = (z_data > 1.2) & (gamma_data < 8.0)
        
        if np.sum(mask) > 2:
            slope = calculate_slope(z_data[mask], gamma_data[mask])
        else:
            slope = 0.0
            
        label_text = f"{name} (Initial Slope: {slope:.2f})"
        ax.plot(z_data, gamma_data, label=label_text, color=model['color'], linestyle=model['ls'], linewidth=2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Lorentz Factor $\Gamma$', fontsize=12)
ax.set_title('Acceleration Profile', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc='upper left') 

# --- Subplot 1b: Jet Shape (Radius vs Z) ---
ax = axes1[1]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    r_data = np.array(results[name]['r']) / R_0
    
    if len(z_data) > 2:
        mask = z_data > 10
        slope = calculate_slope(z_data[mask], r_data[mask])
        b_val = b_values.get(name, 0.0)
        label_text = f"{name} (b={b_val}, Slope: {slope:.2f})"
        ax.plot(z_data, r_data, label=label_text, color=model['color'], linestyle=model['ls'], linewidth=2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Jet Radius $r$ ($R_0$)', fontsize=12)
ax.set_title('Jet Geometry / Collimation', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc='upper left') 

plt.tight_layout()
fig1.savefig(os.path.join(output_dir, "PAIR_1_FILENAME.png"), dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# PAIR 7: 4-Velocity (v * Gamma) and Enthalpy (h)
# ------------------------------------------------------------------------------
fig7, axes7 = plt.subplots(1, 2, figsize=(15, 6))
plt.subplots_adjust(wspace=0.25)
c_speed = 2.9979e10

# --- Subplot 7a: v * Gamma (4-Velocity) ---
ax = axes7[0]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    gamma_data = np.array(results[name]['gamma'])
    
    if len(z_data) > 2:
        # v = c * sqrt(1 - 1/gamma^2) --> v*gamma = c * sqrt(gamma^2 - 1)
        v_gamma = c_speed * np.sqrt(gamma_data**2 - 1.0)
        
        # Calculate slope for z > 10 Z_0
        mask = z_data > 10
        slope = calculate_slope(z_data[mask], v_gamma[mask])
        label_text = f"{name} (Slope: {slope:.2f})"
        
        ax.plot(z_data, v_gamma, label=label_text, color=model['color'], linestyle=model['ls'], linewidth=2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'$V_{bulk} * \Gamma$ (cm/s)', fontsize=12)
ax.set_title('Proper Spatial Velocity ($V_{bulk} * \Gamma$)', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc='upper left')

# --- Subplot 7b: Specific Enthalpy (h) ---
ax = axes7[1]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    gamma_data = np.array(results[name]['gamma'])
    
    if len(z_data) > 2:
        # Bernoulli conservation: h * Gamma = 10
        h_data = 10.0 / gamma_data
        
        # Calculate slope for z > 10 Z_0
        mask = z_data > 10
        slope = calculate_slope(z_data[mask], h_data[mask])
        label_text = f"{name} (Slope: {slope:.2f})"
        
        ax.plot(z_data, h_data, label=label_text, color=model['color'], linestyle=model['ls'], linewidth=2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Specific Enthalpy $h$', fontsize=12)
ax.set_title('Specific Enthalpy Evolution', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc='upper right')

plt.tight_layout()
fig7.savefig(os.path.join(output_dir, "PAIR_7_FILENAME.png"), dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# PAIR 2: Magnetic Field Decay and Thermal-Magnetic Pressure
# ------------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))

# --- Subplot 2a: Magnetic Field Decay (B vs Z) ---
ax = axes2[0]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    B_data = np.array(results[name]['B'])

    if len(z_data) > 2:
        mask = z_data > 10
        slope = calculate_slope(z_data[mask], B_data[mask])
        label_text = f"{name} (Slope: {slope:.2f})"
        ax.plot(z_data, B_data, label=label_text, color=model['color'], linestyle=model['ls'], linewidth=2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Comoving B-Field $B^{\prime}$ (Gauss)', fontsize=12)
ax.set_title('Magnetic Field Decay', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc='upper right') # Fixed Overlap

# --- Subplot 2b: Thermal vs Magnetic Pressure ---
ax = axes2[1]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    P_data = np.array(results[name]['P_z'])
    B_data = np.array(results[name]['B'])

    if len(z_data) > 2:
        mask = z_data > 10
        slope_P = calculate_slope(z_data[mask], P_data[mask])
        label_text = f"{name} (Slope: {slope_P:.2f})"
        ax.plot(z_data, P_data, label=label_text, color=model['color'], linestyle=model['ls'], linewidth=2)
        
        P_mag_data = B_data**2 / (8.0 * np.pi)
        ax.plot(z_data, P_mag_data, color=model['color'], linestyle=model['ls'], linewidth=1.5, alpha=0.6, label=f"{name} $P_{{mag}}$")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Pressure (dyn/cm$^2$)', fontsize=12)
ax.set_title('Thermal vs Magnetic Pressure', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc='upper right') # Fixed Overlap

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, "PAIR_2_FILENAME.png"), dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# PAIR 3: r-z Powerlaw Exponent and Expansion-Sound Speed
# ------------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))

# --- Subplot 3a: r(z) Exponent Profile ---
ax = axes3[0]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    r_data = np.array(results[name]['r']) / R_0
    
    if len(z_data) > 2:
        log_z = np.log10(z_data)
        log_r = np.log10(r_data)
        
        exponent_local = np.zeros_like(z_data)
        for i in range(1, len(z_data) - 1):
            exponent_local[i] = (log_r[i+1] - log_r[i-1]) / (log_z[i+1] - log_z[i-1])
        exponent_local[0] = (log_r[1] - log_r[0]) / (log_z[1] - log_z[0])
        exponent_local[-1] = (log_r[-1] - log_r[-2]) / (log_z[-1] - log_z[-2])
        
        ax.plot(z_data, exponent_local, label=name, color=model['color'], linestyle=model['ls'], linewidth=2)

ax.set_xscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Exponent ($d\log r / d\log z$)', fontsize=12)
ax.set_title(r'r(z) Power-Law Exponent', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc='lower right') # Fixed Overlap (plateaus at 1.0)

# --- Subplot 3b: Sound Speed/Gamma vs dr/dt ---
ax = axes3[1]
for model in models:
    name = model['name']
    z_data_original = np.array(results[name]['z'])
    z_data_plot = z_data_original / Z_0
    gamma_data = np.array(results[name]['gamma'])
    r_data = np.array(results[name]['r'])
    c_s_data = np.array(results[name]['c_s'])  

    if len(r_data) > 2:
        dr_dz = np.zeros_like(r_data)
        for i in range(1, len(r_data) - 1):
            dr_dz[i] = (r_data[i+1] - r_data[i-1]) / (z_data_original[i+1] - z_data_original[i-1])
        dr_dz[0] = (r_data[1] - r_data[0]) / (z_data_original[1] - z_data_original[0])
        dr_dz[-1] = (r_data[-1] - r_data[-2]) / (z_data_original[-1] - z_data_original[-2])
        
        c = 2.9979e10
        v_data = c * np.sqrt(1.0 - 1.0 / gamma_data**2)
        v_perp = v_data * dr_dz
        expansion_ratio = (v_perp * gamma_data / c_s_data) 
        mask = z_data_plot > 10
        slope = calculate_slope(z_data_plot[mask], expansion_ratio[mask])
        label_text = f"{name} (Slope: {slope:.2f})"

        ax.plot(z_data_plot, expansion_ratio, label=label_text, color=model['color'], linestyle=model['ls'], linewidth=2, marker='o', markersize=3, alpha=0.7)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Expansion Speed in flow frame / Sound Speed', fontsize=12)
ax.set_title(r'Expansion Speed in flow frame / Sound Speed', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc='lower center') # Fixed Overlap

plt.tight_layout()
fig3.savefig(os.path.join(output_dir, "PAIR_3_FILENAME.png"), dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# PAIR 4: Energy and Mass Conservation Checks
# ------------------------------------------------------------------------------
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))

# --- Subplot 4a: Energy Evolution ---
ax = axes4[0]
ax.axhline(10**44, color='black', linestyle=':', linewidth=3, alpha=0.8, label=r'Total Power ($L_{tot}$)')

for model in models:
    name = model['name']
    c = model['color']
    z_plot = np.array(results[name]['z']) / Z_0
    l_kin_arr = np.array(results[name]['l_kin'])
    l_th_arr = np.array(results[name]['l_th'])
    
    if len(z_plot) > 0:
        ax.plot(z_plot, l_kin_arr, color=c, linestyle='-', linewidth=2.5, alpha=0.85, label=f"{name} $L_{{kin}}$")
        ax.plot(z_plot, l_th_arr, color=c, linestyle='--', linewidth=2.5, alpha=0.85, label=f"{name} $L_{{th}}$")

ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_xlim(1, 300)
ax.set_ylim(0, 1.2e44) 
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Luminosity (erg/s)', fontsize=12)
ax.set_title(r'Power Evolution (Linear Energy Scale)', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9, loc='lower left', ncol=2)

# --- Subplot 4b: Mass Rate ---
ax = axes4[1]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    M_dot = np.array(results[name]['M_dot'])
    if len(z_data) > 0:
        ax.plot(z_data, M_dot, label=name, color=model['color'], linestyle=model['ls'], linewidth=2, marker='o', markersize=3, alpha=0.7)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
M_dot_base = 10**44 / ((10.0 - 1.0) * (2.9979e10)**2)
ax.set_ylim(M_dot_base * 0.9, M_dot_base * 1.1)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Mass Rate ($\dot{M}$)', fontsize=12)
ax.set_title(r'Mass Rate Conservation Check', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc='center right')

plt.tight_layout()
fig4.savefig(os.path.join(output_dir, "PAIR_4_FILENAME.png"), dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# PAIR 5: Differential Spectral Energy Flux (z * d(nu*F_nu)/dz)
# ------------------------------------------------------------------------------
fig5, axes5 = plt.subplots(1, 2, figsize=(16, 6))
model_b_name = "Model B:"

# Safely get z_data for the correct model name format
if model_b_name in results:
    z_data = np.array(results[model_b_name]['z']) / Z_0
elif "Model B" in results:
    model_b_name = "Model B"
    z_data = np.array(results[model_b_name]['z']) / Z_0
else:
    z_data = []

if len(z_data) > 0:
    num_epochs = 50
    indices = np.linspace(0, len(z_data) - 1, num_epochs, dtype=int)
    cmap = plt.cm.viridis

    # --- Subplot 5a: Theta = 10° ---
    ax = axes5[0]
    F_obs_1_z_nu = np.array(results[model_b_name]['F_obs_1_z_nu'])
    n_c_list_thita1 = np.array(results[model_b_name]['n_c_list_thita1'])

    for j, idx in enumerate(indices):
        color = cmap(j / (num_epochs - 1))
        freq_mask = np.array(n_c_list_thita1[idx]) >= 1e7
        n_c_filtered = np.array(n_c_list_thita1[idx])[freq_mask]
        F_obs_1_filtered = np.array(F_obs_1_z_nu[idx])[freq_mask]
        
        if len(n_c_filtered) > 0:
            # Only assign a label to the first and last lines to avoid a 50-item legend
            label_text = "_nolegend_" 
            if j == 0 or j == (num_epochs - 1):
                if len(n_c_filtered) > 2:
                    slope = calculate_slope(n_c_filtered, F_obs_1_filtered)
                    label_text = f'z={z_data[idx]:.2f} $Z_0$ (Slope: {slope:.2f})'
                else:
                    label_text = f'z={z_data[idx]:.2f} $Z_0$'

            ax.loglog(n_c_filtered, F_obs_1_filtered, marker='.', alpha=0.8, linewidth=1.5, label=label_text, color=color)

    ax.set_xlabel(r'Frequency $\nu$ (Hz)', fontsize=12)
    ax.set_ylabel(r'$z \frac{d(\nu F_{\nu})}{dz}$ (erg/s/cm$^2$)', fontsize=14)
    ax.set_title(r'Differential Spectral Flux per Log-Distance ($\theta=10^{\circ}$)', fontsize=13, fontweight='bold')
    ax.set_xlim(1e7, 1e18)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc='lower left')

    # --- Subplot 5b: Theta = 45° ---
    ax = axes5[1]
    F_obs_2_z_nu = np.array(results[model_b_name]['F_obs_2_z_nu'])
    n_c_list_thita2 = np.array(results[model_b_name]['n_c_list_thita2'])

    for j, idx in enumerate(indices):
        color = cmap(j / (num_epochs - 1))
        freq_mask = np.array(n_c_list_thita2[idx]) >= 1e7
        n_c_filtered = np.array(n_c_list_thita2[idx])[freq_mask]
        F_obs_2_filtered = np.array(F_obs_2_z_nu[idx])[freq_mask]
        
        if len(n_c_filtered) > 0:
            # Only assign a label to the first and last lines to avoid a 50-item legend
            label_text = "_nolegend_" 
            if j == 0 or j == (num_epochs - 1):
                if len(n_c_filtered) > 2:
                    slope = calculate_slope(n_c_filtered, F_obs_2_filtered)
                    label_text = f'z={z_data[idx]:.2f} $Z_0$ (Slope: {slope:.2f})'
                else:
                    label_text = f'z={z_data[idx]:.2f} $Z_0$'

            ax.loglog(n_c_filtered, F_obs_2_filtered, marker='.', alpha=0.8, linewidth=1.5, label=label_text, color=color)

    ax.set_xlabel(r'Frequency $\nu$ (Hz)', fontsize=12)
    ax.set_ylabel(r'$z \frac{d(\nu F_{\nu})}{dz}$ (erg/s/cm$^2$)', fontsize=14)
    ax.set_title(r'Differential Spectral Flux per Log-Distance ($\theta=45^{\circ}$)', fontsize=13, fontweight='bold')
    ax.set_xlim(1e7, 1e18)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc='lower left')

plt.tight_layout()
fig5.savefig(os.path.join(output_dir, "PAIR_5_FILENAME.png"), dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# PAIR 6: Spectral Energy Flux per Distance (UNIFIED FREQUENCIES)
# ------------------------------------------------------------------------------
fig6, axes6 = plt.subplots(1, 2, figsize=(16, 6))
cmap = plt.cm.plasma
unified_target_frequencies = [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]

if len(z_data) > 0:
    # --- Subplot 6a: Flux vs Z - Theta 10° ---
    ax = axes6[0]
    for j, target_freq in enumerate(unified_target_frequencies):
        color = cmap(j / max(len(unified_target_frequencies) - 1, 1))
        F_obs_1_z_vs_z = []
        for z_idx in range(len(z_data)):
            freq_array_at_z = np.array(n_c_list_thita1[z_idx])
            flux_array_at_z = np.array(F_obs_1_z_nu[z_idx]) 
            # FIXED: Return NaN if the frequency is physically unachievable
            interpolated_flux = np.interp(target_freq, freq_array_at_z, flux_array_at_z, left=np.nan, right=np.nan)
            F_obs_1_z_vs_z.append(interpolated_flux)
        ax.loglog(z_data, F_obs_1_z_vs_z, marker='o', markersize=4, alpha=0.8, linewidth=2, label=f'$\\nu =$ {target_freq:.0e} Hz', color=color)

    ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
    ax.set_ylabel(r'$z \frac{d(\nu F_{\nu})}{dz}$ (erg/s/cm$^2$)', fontsize=14)
    ax.set_title(r'$\nu F_\nu$ Evolution Along Jet - Theta=10°', fontsize=14)
    ax.set_xlim(1, 300)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc='lower left')

    # --- Subplot 6b: Flux vs Z - Theta 45° ---
    ax = axes6[1]
    for j, target_freq in enumerate(unified_target_frequencies):
        color = cmap(j / max(len(unified_target_frequencies) - 1, 1))
        F_obs_2_z_vs_z = []
        for z_idx in range(len(z_data)):
            freq_array_at_z = np.array(n_c_list_thita2[z_idx])
            flux_array_at_z = np.array(F_obs_2_z_nu[z_idx]) 
            # FIXED: Return NaN if the frequency is physically unachievable
            interpolated_flux = np.interp(target_freq, freq_array_at_z, flux_array_at_z, left=np.nan, right=np.nan)
            F_obs_2_z_vs_z.append(interpolated_flux)
        ax.loglog(z_data, F_obs_2_z_vs_z, marker='o', markersize=4, alpha=0.8, linewidth=2, label=f'$\\nu =$ {target_freq:.0e} Hz', color=color)

    ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
    ax.set_ylabel(r'$z \frac{d(\nu F_{\nu})}{dz}$ (erg/s/cm$^2$)', fontsize=14)
    ax.set_title(r'$\nu F_\nu$ Evolution Along Jet - Theta=45° ', fontsize=14)
    ax.set_xlim(1, 300)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc='lower left')

plt.tight_layout()
fig6.savefig(os.path.join(output_dir, "PAIR_6_FILENAME.png"), dpi=300, bbox_inches='tight')
plt.show()
# ------------------------------------------------------------------------------
# FINAL 3: TOTAL SED (INTEGRATED OVER Z) [PROFESSOR REQUEST]
# ------------------------------------------------------------------------------
fig_final3 = plt.figure(figsize=(10, 6))

if len(z_data) > 0:
    # Create a common frequency grid to interpolate onto
    nu_global = np.logspace(7, 18, 500)
    
    integrand_1 = []
    integrand_2 = []
    
    # Interpolate the differential flux onto the global frequency grid for each z-slice
    for i in range(len(z_data)):
        nu1 = np.array(n_c_list_thita1[i])
        f1 = np.array(F_obs_1_z_nu[i])
        # Fill with 0 if outside the bounds of the local frequency array
        int_1 = np.interp(nu_global, nu1, f1, left=0, right=0)
        integrand_1.append(int_1)
        
        nu2 = np.array(n_c_list_thita2[i])
        f2 = np.array(F_obs_2_z_nu[i])
        int_2 = np.interp(nu_global, nu2, f2, left=0, right=0)
        integrand_2.append(int_2)
        
    integrand_1 = np.array(integrand_1)
    integrand_2 = np.array(integrand_2)
    
    # Integrate mathematically over d(ln z) using the trapezoidal rule
    # Integral of [z * dF/dz] d(ln z) = Integral of dF = Total Flux
    log_z_data = np.log(z_data)
    total_sed_1 = np.trapz(integrand_1, x=log_z_data, axis=0)
    total_sed_2 = np.trapz(integrand_2, x=log_z_data, axis=0)
    
    plt.loglog(nu_global, total_sed_1, linewidth=3, label=r'$\theta = 10^{\circ}$', color='#e74c3c')
    plt.loglog(nu_global, total_sed_2, linewidth=3, label=r'$\theta = 45^{\circ}$', color='#3498db')

    plt.xlabel(r'Frequency $\nu$ (Hz)', fontsize=12)
    plt.ylabel(r'Total Observed SED $\nu F_{\nu}(z/dz)$ (erg/s/cm$^2$)', fontsize=12)
    plt.title('Total Integrated Spectral Energy Distribution (Observer)', fontsize=14, fontweight='bold')
    plt.xlim(1e7, 1e18)
    
    # Set bottom y-limit dynamically to ignore 0 artifacts from interpolation
    valid_y = np.concatenate([total_sed_1[total_sed_1 > 0], total_sed_2[total_sed_2 > 0]])
    if len(valid_y) > 0:
        plt.ylim(np.min(valid_y) * 0.1, np.max(valid_y) * 10)
        
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=12, loc='upper right')

plt.tight_layout()
fig_final3.savefig(os.path.join(output_dir, "FINAL_3_SED_FILENAME.png"), dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# FINAL 1: Total Synchrotron Flux
# ------------------------------------------------------------------------------
fig_final1 = plt.figure(figsize=(10, 6))

if len(z_data) > 0:
    F_bol_obs_1 = np.array(results[model_b_name]['F_bol_obs_1']) 
    F_bol_obs_2 = np.array(results[model_b_name]['F_bol_obs_2']) 
    
    # Filter for valid data
    valid_mask1 = np.isfinite(F_bol_obs_1) & (F_bol_obs_1 > 0)
    valid_mask2 = np.isfinite(F_bol_obs_2) & (F_bol_obs_2 > 0)
    
    # Mask for stable regime slope calculation (z > 10)
    slope_mask1 = valid_mask1 & (z_data > 10)
    slope_mask2 = valid_mask2 & (z_data > 10)

    # Slope for Theta = 10
    if np.sum(slope_mask1) > 2:
        slope_1 = calculate_slope(z_data[slope_mask1], F_bol_obs_1[slope_mask1])
        label_1 = r'$\theta = 10^{\circ}$ (Slope: %.2f)' % slope_1
    else:
        label_1 = r'$\theta = 10^{\circ}$'

    # Slope for Theta = 45
    if np.sum(slope_mask2) > 2:
        slope_2 = calculate_slope(z_data[slope_mask2], F_bol_obs_2[slope_mask2])
        label_2 = r'$\theta = 45^{\circ}$ (Slope: %.2f)' % slope_2
    else:
        label_2 = r'$\theta = 45^{\circ}$'

    plt.loglog(z_data[valid_mask1], F_bol_obs_1[valid_mask1], marker='o', markersize=4, alpha=0.8, linewidth=2.5, label=label_1, color='#e74c3c')
    plt.loglog(z_data[valid_mask2], F_bol_obs_2[valid_mask2], marker='s', markersize=4, alpha=0.8, linewidth=2.5, label=label_2, color='#3498db')

    plt.xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
    # Note: Updated label to reflect the mathematically scaled z*dF/dz
    plt.ylabel(r'Total Synchrotron Flux $z \frac{dF_{\mathrm{sync}}}{dz}$ (erg/s/cm$^2$)', fontsize=12)
    plt.title('Total Synchrotron Flux vs Distance (As Seen by Observer)', fontsize=14, fontweight='bold')
    plt.xlim(1, 300)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=12, loc='upper right') 
    
plt.tight_layout()
fig_final1.savefig(os.path.join(output_dir, "FINAL_1_FILENAME.png"), dpi=300, bbox_inches='tight')
plt.show()
# ==============================================================================
# FINAL 2: RADIATIVE EFFICIENCY & APPARENT LUMINOSITY (Scaled z*dL/dz)
# ==============================================================================
print("\n" + "="*70)
print("RADIATIVE EFFICIENCY CHECK (Differential Synchrotron Luminosity vs Jet Power)")
print("="*70)

fig_final2, axes_f2 = plt.subplots(1, 3, figsize=(16, 5))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

for idx, model in enumerate(models):
    name = model['name']
    ax = axes_f2[idx]
    
    if 'L_lab_scaled' in results[name] and len(results[name]['L_lab_scaled']) > 0:
        z_data_rad = np.array(results[name]['z']) / Z_0
        L_lab_scaled_array = np.array(results[name]['L_lab_scaled'])
        valid_mask = np.isfinite(L_lab_scaled_array)
        
        l_total_array = np.array(results[name]['l_total'])
        Total_Jet_Power = l_total_array[np.isfinite(l_total_array)][0]
        
        # 1. Total Jet Power (Solid Black)
        ax.axhline(Total_Jet_Power, color='black', linestyle='-', linewidth=2.5, label='Total Jet Power ($L_j$)')
        
        # 2. 1% Threshold (Dashed Gray)
        threshold_1percent = 0.01 * Total_Jet_Power
        ax.axhline(threshold_1percent, color='gray', linestyle='--', linewidth=2, label='1% Threshold')
        
        # --- SLOPE CALCULATION MASK (Stable Regime z > 10) ---
        slope_mask = valid_mask & (z_data_rad > 10)
        
        # 3. Intrinsic Lab Luminosity Scaled (Distinct Blue / Circle Marker)
        if np.sum(slope_mask) > 2:
            slope_lab = calculate_slope(z_data_rad[slope_mask], L_lab_scaled_array[slope_mask])
            label_lab = r'Intrinsic $z \frac{dL_{\mathrm{lab}}}{dz}$ (Slope: %.2f)' % slope_lab
        else:
            label_lab = r'Intrinsic $z \frac{dL_{\mathrm{lab}}}{dz}$'
            
        ax.loglog(z_data_rad[valid_mask], L_lab_scaled_array[valid_mask], marker='o', markersize=4, linewidth=2, 
                    label=label_lab, color='#3498db', alpha=0.8)
        
        # 4. Apparent Observer Luminosity at Theta = 10° (Crimson Red / Triangle Marker)
        if 'L_bol_app_1_scaled' in results[name] and len(results[name]['L_bol_app_1_scaled']) == len(z_data_rad):
            L_app_1_scaled = np.array(results[name]['L_bol_app_1_scaled'])
            
            if np.sum(slope_mask) > 2:
                slope_app = calculate_slope(z_data_rad[slope_mask], L_app_1_scaled[slope_mask])
                label_app = r'Apparent $z \frac{dL_{\mathrm{app}}}{dz}$ ($\theta=10^{\circ}$) (Slope: %.2f)' % slope_app
            else:
                label_app = r'Apparent $z \frac{dL_{\mathrm{app}}}{dz}$ ($\theta=10^{\circ}$)'
                
            ax.loglog(z_data_rad[valid_mask], L_app_1_scaled[valid_mask], marker='^', markersize=4, linewidth=2, 
                        label=label_app, color='#e74c3c', alpha=0.9)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 300)
        ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=11)
        ax.set_ylabel(r'Differential Luminosity (erg/s)', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, which="both", alpha=0.3)
        
        ax.legend(fontsize=9, loc='lower left')
        
        max_lum = np.max(L_lab_scaled_array[valid_mask])
        max_ratio = max_lum / Total_Jet_Power * 100
        
        print(f"\n{name}:")
        print(f"  Total Jet Power (L_j):         {Total_Jet_Power:.4e} erg/s")
        print(f"  Max Intrinsic L_lab (z/dz):    {max_lum:.4e} erg/s ({max_ratio:.4f}% of L_j)")
    else:
        print(f"\n{name}: No valid scaled data generated to plot.")

plt.tight_layout()
fig_final2.savefig(os.path.join(output_dir, "FINAL_2_FILENAME.png"), dpi=300, bbox_inches='tight')
plt.show()
# ==============================================================================
# TERMINAL CONSERVATION PRINTS
# ==============================================================================
print("\n" + "="*70)
print("CONSERVATION CHECKS: Mass Rate and Energy Rate (without B-field)")
print("="*70)

for model in models:
    name = model['name']
    M_dot_c = np.array(results[name]['M_dot'])
    L_no_mag = np.array(results[name]['L_no_mag'])
    
    print(f"\n{name}:")
    if len(M_dot_c) > 1 and np.max(np.abs(M_dot_c)) > 0:
        M_dot_mean = np.mean(np.abs(M_dot_c))
        M_dot_std = np.std(M_dot_c)
        M_dot_variation = M_dot_std / M_dot_mean
        print(f"  Mass Rate Conservation (ρ·A·v·Γ = constant):")
        print(f"    Value: {M_dot_mean:.4e} g/s")
        print(f"    Relative Variation: {M_dot_variation:.4e} ({M_dot_variation*100:.2f}%)")
    
    if len(L_no_mag) > 1 and np.max(np.abs(L_no_mag)) > 0:
        L_no_mag_mean = np.mean(np.abs(L_no_mag))
        L_no_mag_std = np.std(L_no_mag)
        L_no_mag_variation = L_no_mag_std / L_no_mag_mean
        print(f"  Energy Rate Conservation (L_kin + L_th = constant):")
        print(f"    Value: {L_no_mag_mean:.4e} erg/s")
        print(f"    Relative Variation: {L_no_mag_variation:.4e} ({L_no_mag_variation*100:.2f}%)")
print("\n" + "="*70)
