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

        if not self.conical_locked and self.z_prev is not None and self.r_prev is not None:
            dr_dz = (radius_test - self.r_prev) / (z - self.z_prev)

            if dr_dz >= theta_M:   # jet opening angle assumed to be small so tan(thetaj) approx thetaj 
                self.conical_locked = True
                self.r_z_lock = self.r_prev / self.z_prev

        if self.conical_locked:
            radius = (self.r_prev / self.z_prev) * z
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
        l_total = l_kin + l_th + l_mag

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

        # Radiation Spectrum per Bin
        # Electron internal energy fraction per bin, flow frame
        dz_array = np.gradient(z_values)
        d = 30 * 3.086e21  # Distance in cm
        area_constant = 4.0 * np.pi * d**2.0
        # Find the index of current z using binary search (much faster than argmin)
        z_index = np.searchsorted(z_values, z)
        z_index = min(z_index, len(z_values) - 1)  # Clamp to valid range
        dz = dz_array[z_index]
        bita = v / self.c
        eta = 0.1
        U_int = 3.0 * P_z
        U_e_cell = eta * U_int * dz * np.pi * radius**2
        m_e = 9.10938356e-28
        E_e_min = m_e * self.c**2.0
        E_e_max = 1e6 * m_e * self.c**2.0 * (rho / self.rho0)**(1.0 / 3.0)
        sigma_T = 6.6524587158e-25
        N_o = U_e_cell / (np.log(E_e_max / E_e_min))
        q_e = 4.80320425e-10  # Statcoulombs (esu)
        E_values = np.geomspace(E_e_min, E_e_max, 500)
        n_c_list_thita1 = []
        n_c_list_thita2 = []
        nLn_ff_list = []
        nLn_app_list_thita1 = []
        nLn_app_list_thita2 = []
        F_obs_1 = []
        F_obs_2 = []
        for i in range(len(E_values)):
            E_e_1 = E_values[i]
            gamma_e = E_e_1 / (m_e * self.c**2.0)
            bita_e = np.sqrt(1.0 - 1.0 / gamma_e**2.0)
            DEradDt = (4.0 / 3.0) * sigma_T * bita_e**2.0 * self.c * (B_prime**2.0 / (8.0 * np.pi)) * (E_e_1 / (m_e * self.c**2.0))**2.0
            N_e = N_o * (E_e_1**-2.0)
            nLn = 0.5 * E_e_1 * N_e * DEradDt  # flow frame
            n_c = 3.0 / (4.0 * np.pi) * q_e * B_prime * E_e_1**2.0 / (m_e**3.0 * self.c**5.0)
            nLn_ff_list.append(nLn)  # remember step is logarithmic
            # now in the observers frame, doppler
            # defining range of frequencies
            n_c_min = 3.0 / (4.0 * np.pi) * q_e * B_prime * E_e_min**2.0 / (m_e**3.0 * self.c**5.0)
            n_c_max = 3.0 / (4.0 * np.pi) * q_e * B_prime * E_e_max**2.0 / (m_e**3.0 * self.c**5.0)
            D_n = (np.log(n_c_max) - np.log(n_c_min)) / 499  # freq step in log space

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

        F_obs_1 = Ln_array_1 / (nu_array_1 * area_constant) * 10**26  # Flux in erg/s/cm^2/Hz, converted to mJy
        F_obs_2 = Ln_array_2 / (nu_array_2 * area_constant) * 10**26  # Flux in erg/s/cm^2
        F_obs_1_z = F_obs_1 * z / dz
        F_obs_2_z = F_obs_2 * z / dz
        F_bol_obs_1 = L_bol_app_1 / area_constant * 10**26
        F_bol_obs_2 = L_bol_app_2 / area_constant * 10**26

        return {'z': z, 'gamma': gamma, 'radius': radius, 'B_prime': B_prime, 'n_c_list_thita1': n_c_list_thita1, 'nLn_app_list_thita1': nLn_app_list_thita1,
                'n_c_list_thita2': n_c_list_thita2, 'nLn_app_list_thita2': nLn_app_list_thita2,
                'P_z': P_z, 'l_kin': l_kin, 'l_th': l_th, 'l_mag': l_mag,
                'l_total': l_total, 'M_dot': M_dot, 'L_no_mag': L_no_mag, 'c_s': c_s, 'c_s_lab': c_s_lab, 'dz_array': dz_array, 'L_lab': L_lab, 'L_bol_app_2': L_bol_app_2, 'L_bol_app_1': L_bol_app_1, 'F_obs_1': F_obs_1, 'F_obs_2': F_obs_2, 'F_obs_1_z': F_obs_1_z, 'F_obs_2_z': F_obs_2_z, 'F_bol_obs_1': F_bol_obs_1, 'F_bol_obs_2': F_bol_obs_2}
# ==============================================================================
# 3. DEFINE SIMULATION PARAMETERS
# ==============================================================================

c = 2.9979e10
V_0 = c / np.sqrt(3.0) 
gamma_0 = 1.0 / np.sqrt(1.0 - (V_0/c)**2)  
h_gamma = 10.0
h_0 = h_gamma / gamma_0  # Will be ~8.165
R_0 = 3.086e20  # 100 parsecs in cm
M_dot = 10**44 / ((h_gamma-1)*c**2)  # Mass rate corresponding to a kinetic luminosity of 10^44 erg/s at the base
RHO_0 = M_dot / (V_0*gamma_0*np.pi*R_0**2)
P_0 = (h_0 - 1.0) * (RHO_0 * c**2) / 4.0
B_0 =   np.sqrt(0.01*8.0 * np.pi *3 * P_0)  # 1% of the thermal pressure in Gauss
Z_0 = 3.086e21  # 1 kiloparsec in centimeters
Z_scale = Z_0

models = [
    {"name": "Model A:",      "z_scale":Z_0, "exponent": 8.0/3.0,"color": "#e74c3c", "ls": ":"},
    {"name": "Model B:", "z_scale":Z_0, "exponent": 4, "color": "#2ecc71", "ls": "-"},
    {"name": "Model C:",   "z_scale":Z_0, "exponent": 2.0, "color": "#3498db", "ls": "--"},
]

# ==============================================================================
# 4. RUN SIMULATIONS
# ==============================================================================

results = {m['name']: {'z': [], 'gamma': [], 'r': [], 'B': [], 'P_z': [], 'l_kin': [], 'l_th': [], 'l_mag': [], 'l_total': [], 'M_dot': [], 'L_no_mag': [], 'c_s': [], 'c_s_lab': [], 'n_c_list_thita1': [], 'nLn_app_list_thita1': [], 'n_c_list_thita2': [], 'nLn_app_list_thita2': [], 'L_lab': [], 'L_bol_app_1': [], 'L_bol_app_2': [], 'F_obs_1': [], 'F_obs_2': [], 'F_obs_1_z': [], 'F_obs_2_z': [], 'F_bol_obs_1': [], 'F_bol_obs_2': []} for m in models}

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
            # Debug: print first 5 radii for Model B
            if model['name'] == "Model A:" and len(results[model['name']]['r']) <= 5:
                print(f"Model A: z={z:.2e} cm, radius={data['radius']:.2e} cm")


# ==============================================================================
# 5. PLOTTING
# ==============================================================================

print("\nGenerating Plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# --- PLOT 1: Acceleration Profile (Gamma vs Z) ---
ax = axes[0, 0]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    gamma_data = results[name]['gamma']
    
    # Calculate slope for gamma decay (for z > 10 Z_0)
    mask = z_data > 10
    slope = calculate_slope(z_data[mask], np.array(gamma_data)[mask])
    
    label_text = f"{name} (Slope: {slope:.2f})"
    
    ax.plot(z_data, gamma_data,
            label=label_text, color=model['color'], linestyle=model['ls'], linewidth=2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Lorentz Factor $\Gamma$', fontsize=12)
ax.set_title('Acceleration Profile', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend()

# --- PLOT 2: Jet Shape (Radius vs Z) ---
ax = axes[0, 1]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    r_data = np.array(results[name]['r']) / R_0
    ax.plot(z_data, r_data,
            label=name, color=model['color'], linestyle=model['ls'], linewidth=2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Jet Radius $r$ ($R_0$)', fontsize=12)
ax.set_title('Jet Geometry / Collimation', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)

# --- PLOT 3: Magnetic Field Decay (B vs Z) with SLOPES ---
ax = axes[1, 0]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    B_data = results[name]['B']

    # Calculate slope for B-field decay (for z > 10 Z_0)
    mask = z_data > 10
    slope = calculate_slope(z_data[mask], np.array(B_data)[mask])
    
    label_text = f"{name} (Slope: {slope:.2f})"

    ax.plot(z_data, B_data,
            label=label_text, color=model['color'], linestyle=model['ls'], linewidth=2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Comoving B-Field $B^{\prime}$ (Gauss)', fontsize=12)
ax.set_title('Magnetic Field Decay', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend()

# --- PLOT 4: Thermal Pressure Decay (P_th vs Z) with SLOPES ---
ax = axes[1, 1]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    P_data = results[name]['P_z']
    B_data = results[name]['B']

    # Calculate slope for Pressure decay (for z > 10 Z_0)
    mask = z_data > 10
    slope_P = calculate_slope(z_data[mask], np.array(P_data)[mask])

    label_text = f"{name} (Slope: {slope_P:.2f})"

    ax.plot(z_data, P_data,
            label=label_text, color=model['color'], linestyle=model['ls'], linewidth=2)
    
    # Add magnetic pressure: P_mag = B^2 / (8π)
    P_mag_data = np.array(B_data)**2 / (8.0 * np.pi)
    ax.plot(z_data, P_mag_data, color=model['color'], linestyle=model['ls'], 
            linewidth=1.5, alpha=0.6, label=f"{name} $P_{{mag}}$")

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Pressure (dyn/cm$^2$)', fontsize=12)
ax.set_title('Thermal vs Magnetic Pressure', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=10)

plt.show()

# --- PLOT 5: Energy Evolution for Paraboloidal Model (Model B) ---
paraboloidal_results = results["Model B:"]
z_plot = np.array(paraboloidal_results['z']) / Z_0
l_kin_arr = np.array(paraboloidal_results['l_kin'])
l_th_arr = np.array(paraboloidal_results['l_th'])
l_mag_arr = np.array(paraboloidal_results['l_mag'])

if 'l_total' in paraboloidal_results and len(paraboloidal_results['l_total']) == len(z_plot):
    l_total_arr = np.array(paraboloidal_results['l_total'])
else:
    l_total_arr = l_kin_arr + l_th_arr + l_mag_arr

plt.figure(figsize=(10, 7))

finite_mask = np.isfinite(z_plot) & np.isfinite(l_kin_arr) & np.isfinite(l_th_arr) & np.isfinite(l_mag_arr) & np.isfinite(l_total_arr)
if not np.any(finite_mask):
    z_plot = np.array(paraboloidal_results['z'])
    l_kin_arr = np.array(paraboloidal_results['l_kin'])
    l_th_arr = np.array(paraboloidal_results['l_th'])
    l_mag_arr = np.array(paraboloidal_results['l_mag'])
    l_total_arr = l_kin_arr + l_th_arr + l_mag_arr
else:
    z_plot = z_plot[finite_mask]
    l_kin_arr = l_kin_arr[finite_mask]
    l_th_arr = l_th_arr[finite_mask]
    l_mag_arr = l_mag_arr[finite_mask]
    l_total_arr = l_total_arr[finite_mask]

all_vals = np.concatenate([l_kin_arr, l_th_arr, l_mag_arr, l_total_arr])
if all_vals.size > 0:
    lo, hi = np.nanpercentile(all_vals[np.isfinite(all_vals)], [0.5, 99.5])
    hi = max(hi, lo * 1.1 + 1e-30)
else:
    lo, hi = None, None

plt.plot(z_plot, l_kin_arr, label=r'Kinetic Power ($L_{kin}$)', color='blue', linewidth=2)
plt.plot(z_plot, l_th_arr, label=r'Thermal Power ($L_{th}$)', color='red', linewidth=2, linestyle='--')
plt.plot(z_plot, l_mag_arr, label=r'Magnetic Power ($L_{mag}$)', color='green', linewidth=2, linestyle='-.')
plt.plot(z_plot, l_total_arr, label=r'Total Power ($L_{tot}$)', color='black', linewidth=3)

plt.xscale('log')
plt.yscale('linear')
if hi is not None:
    plt.ylim(bottom=0, top=hi * 1.2)
plt.xlim(1, 300)
plt.xlabel(r'Distance $z$ ($Z_0$)', fontsize=14)
plt.ylabel(r'Luminosity (erg/s)', fontsize=14)
plt.title(r'Jet Energy Evolution ( Model)', fontsize=16, fontweight='bold')
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=12)

# ==============================================================================
# PLOT 6: Power Evolution in Log-Log Space + Exponent + Conservation Checks
# ==============================================================================
print("\nGenerating additional plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# --- PLOT 6a: Power Evolution Log-Log (Model B) ---
ax = axes[0, 0]
ax.plot(z_plot, l_kin_arr, label=r'Kinetic Power ($L_{kin}$)', color='blue', linewidth=2)
ax.plot(z_plot, l_th_arr, label=r'Thermal Power ($L_{th}$)', color='red', linewidth=2, linestyle='--')
ax.plot(z_plot, l_mag_arr, label=r'Magnetic Power ($L_{mag}$)', color='green', linewidth=2, linestyle='-.')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Luminosity (erg/s)', fontsize=12)
ax.set_title(r'Power Evolution Log-Log Scale (Model B)', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend()

# --- PLOT 6b: r(z) Exponent Profile ---
ax = axes[0, 1]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    r_data = np.array(results[name]['r']) / R_0
    
    log_z = np.log10(z_data)
    log_r = np.log10(r_data)
    
    exponent_local = np.zeros_like(z_data)
    for i in range(1, len(z_data) - 1):
        exponent_local[i] = (log_r[i+1] - log_r[i-1]) / (log_z[i+1] - log_z[i-1])
    exponent_local[0] = (log_r[1] - log_r[0]) / (log_z[1] - log_z[0])
    exponent_local[-1] = (log_r[-1] - log_r[-2]) / (log_z[-1] - log_z[-2])
    
    ax.plot(z_data, exponent_local, label=name, color=model['color'], 
            linestyle=model['ls'], linewidth=2)

ax.set_xscale('log')
ax.set_xlim(1, 300)
ax.set_ylabel(r'Exponent ($d\log r / d\log z$)', fontsize=12)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_title(r'r(z) Power-Law Exponent', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend()

# --- PLOT 6c: Mass Rate Conservation (excluding B-field contribution) ---
ax = axes[1, 0]
for model in models:
    name = model['name']
    z_data = np.array(results[name]['z']) / Z_0
    M_dot = np.array(results[name]['M_dot'])
    
    ax.plot(z_data, M_dot, label=name, color=model['color'], 
            linestyle=model['ls'], linewidth=2, marker='o', markersize=3, alpha=0.7)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Mass Rate ($\dot{M}$)', fontsize=12)
ax.set_title(r'Mass Rate Conservation Check', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend()

# --- PLOT 6d: Sound Speed/Gamma vs dr/dt ---
ax = axes[1, 1]
for model in models:
    name = model['name']
    z_data_original = np.array(results[name]['z'])  # Keep original for derivatives
    z_data_plot = z_data_original / Z_0  # Normalized for plotting
    gamma_data = np.array(results[name]['gamma'])
    r_data = np.array(results[name]['r'])
    c_s_data = np.array(results[name]['c_s'])
    c_s_lab_data = np.array(results[name]['c_s_lab'])  
  

    #derivative calculation for dr/dz using ORIGINAL z values
    dr_dz = np.zeros_like(r_data)
    for i in range(1, len(r_data) - 1):
        dr_dz[i] = (r_data[i+1] - r_data[i-1]) / (z_data_original[i+1] - z_data_original[i-1])
    dr_dz[0] = (r_data[1] - r_data[0]) / (z_data_original[1] - z_data_original[0])
    dr_dz[-1] = (r_data[-1] - r_data[-2]) / (z_data_original[-1] - z_data_original[-2])
    
# Calculate actual velocity (v) from gamma
    c = 2.9979e10
    v_data = c * np.sqrt(1.0 - 1.0 / gamma_data**2)
    
    # Lateral expansion velocity in the lab frame (v_perp = v * dr/dz)
    v_perp = v_data * dr_dz
    
    # Calculate the true ratio: Expansion Speed / Lab Sound Speed
    expansion_ratio = (v_perp * gamma_data / c) * np.sqrt(3.0)
    #expansion_ratio = (v_perp * gamma_data) / c_s_data

    ax.plot(z_data_plot, expansion_ratio, label=name, color=model['color'], 
            linestyle=model['ls'], linewidth=2, marker='o', markersize=3, alpha=0.7)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 300)
ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
ax.set_ylabel(r'Expansion Speed in flow frame / C', fontsize=12)
ax.set_title(r'Expansion Speed in flow frame / C', fontsize=14, fontweight='bold')
ax.grid(True, which="both", alpha=0.3)
ax.legend()

plt.show()

# ==============================================================================
# CONSERVATION CHECKS: Mass Rate and Energy Rate (excluding B)
# ==============================================================================
print("\n" + "="*70)
print("CONSERVATION CHECKS: Mass Rate and Energy Rate (without B-field)")
print("="*70)

for model in models:
    name = model['name']
    z_data = np.array(results[name]['z'])
    M_dot = np.array(results[name]['M_dot'])
    L_no_mag = np.array(results[name]['L_no_mag'])
    
    print(f"\n{name}:")
    
    if len(M_dot) > 1 and np.max(np.abs(M_dot)) > 0:
        M_dot_mean = np.mean(np.abs(M_dot))
        M_dot_std = np.std(M_dot)
        M_dot_variation = M_dot_std / M_dot_mean
        M_dot_min = np.min(M_dot)
        M_dot_max = np.max(M_dot)
        print(f"  Mass Rate Conservation (ρ·A·v·Γ = constant):")
        print(f"    Value: {M_dot_mean:.4e} g/s")
        print(f"    Std Dev: {M_dot_std:.4e}")
        print(f"    Relative Variation: {M_dot_variation:.4e} ({M_dot_variation*100:.2f}%)")
        print(f"    Range: [{M_dot_min:.4e}, {M_dot_max:.4e}]")
    
    if len(L_no_mag) > 1 and np.max(np.abs(L_no_mag)) > 0:
        L_no_mag_mean = np.mean(np.abs(L_no_mag))
        L_no_mag_std = np.std(L_no_mag)
        L_no_mag_variation = L_no_mag_std / L_no_mag_mean
        L_no_mag_min = np.min(L_no_mag)
        L_no_mag_max = np.max(L_no_mag)
        print(f"  Energy Rate Conservation (L_kin + L_th = constant):")
        print(f"    Value: {L_no_mag_mean:.4e} erg/s")
        print(f"    Std Dev: {L_no_mag_std:.4e} erg/s")
        print(f"    Relative Variation: {L_no_mag_variation:.4e} ({L_no_mag_variation*100:.2f}%)")
        print(f"    Range: [{L_no_mag_min:.4e}, {L_no_mag_max:.4e}] erg/s")

print("\n" + "="*70)

# ==============================================================================    
# EXTRA PLOT: Synchrotron Spectrum for a Jet (Model B) - 50 EPOCHS - THETA 1
# ==============================================================================
plt.figure(figsize=(12, 7))
model_b_name = "Model B:"
z_data = np.array(results[model_b_name]['z']) / Z_0
F_obs_1_z = np.array(results[model_b_name]['F_obs_1_z'])
n_c_list_thita1 = np.array(results[model_b_name]['n_c_list_thita1'])

num_epochs = 50
indices = np.linspace(0, len(z_data) - 1, num_epochs, dtype=int)
cmap = plt.cm.viridis

for j, idx in enumerate(indices):
    color = cmap(j / (num_epochs - 1))
    freq_mask = np.array(n_c_list_thita1[idx]) >= 1e7
    n_c_filtered = np.array(n_c_list_thita1[idx])[freq_mask]
    F_obs_1_filtered = np.array(F_obs_1_z[idx])[freq_mask]
    if len(n_c_filtered) > 0:
        plt.loglog(n_c_filtered, F_obs_1_filtered, marker='.', alpha=0.8, linewidth=1.5, 
                   label=f'z={z_data[idx]:.2f} $Z_0$', color=color)

plt.xlabel(r'Frequency $\nu$ (Hz)', fontsize=12)
# FIX: Removed the double subscript _obs_1_z to avoid LaTeX ValueError, fixed spelling to "received"
plt.ylabel(r'Flux received $F_{\nu, obs}$ (mJy)', fontsize=12)
plt.title('Synchrotron Flux Spectrum Evolution - Theta=10° (50 Epochs)', fontsize=14)
plt.xlim(1e7, 1e16)
plt.text(0.98, 0.02, 'Model B - Observer Frame', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=9, loc='best', ncol=2)
plt.tight_layout()
plt.show()

# ==============================================================================    
# EXTRA PLOT: Synchrotron Flux Spectrum for a Jet (Model B) - 50 EPOCHS - THETA 2
# ==============================================================================
plt.figure(figsize=(12, 7))
model_b_name = "Model B:"
z_data = np.array(results[model_b_name]['z']) / Z_0
n_c_list_thita2 = np.array(results[model_b_name]['n_c_list_thita2'])
F_obs_2_z = np.array(results[model_b_name]['F_obs_2_z'])

num_epochs = 50
indices = np.linspace(0, len(z_data) - 1, num_epochs, dtype=int)
cmap = plt.cm.viridis

for j, idx in enumerate(indices):
    color = cmap(j / (num_epochs - 1))
    freq_mask = np.array(n_c_list_thita2[idx]) >= 1e7
    n_c_filtered = np.array(n_c_list_thita2[idx])[freq_mask]
    F_obs_2_filtered = np.array(F_obs_2_z[idx])[freq_mask]
    if len(n_c_filtered) > 0:
        plt.loglog(n_c_filtered, F_obs_2_filtered, marker='.', alpha=0.8, linewidth=1.5, 
                   label=f'z={z_data[idx]:.2f} $Z_0$', color=color)

plt.xlabel(r'Frequency $\nu$ (Hz)', fontsize=12)
plt.ylabel(r'Flux received $F_{\nu, obs}$ (mJy)', fontsize=12)
plt.title('Synchrotron Flux Spectrum Evolution - Theta=45° (50 Epochs)', fontsize=14)
plt.xlim(1e7, 1e16)
plt.text(0.98, 0.02, 'Model B  - Observer Frame', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=9, loc='best', ncol=2)
plt.tight_layout()
plt.show()


# ==============================================================================    
# EXTRA PLOT: F_obs_1_z vs Z for different frequencies - THETA 1
# ==============================================================================
plt.figure(figsize=(12, 7))
model_b_name = "Model B:"
z_data = np.array(results[model_b_name]['z']) / Z_0
n_c_list_thita1 = np.array(results[model_b_name]['n_c_list_thita1'])
F_obs_1_z = np.array(results[model_b_name]['F_obs_1_z'])

# Define specific TARGET frequencies to track like a real telescope
target_frequencies = [1e7,1e8, 1e9, 1e10,1e11, 1e12,1e13, 1e14,1e15, 1e16]
cmap = plt.cm.plasma

for j, target_freq in enumerate(target_frequencies):
    color = cmap(j / max(len(target_frequencies) - 1, 1))
    F_obs_1_z_vs_z = []
    
    for z_idx in range(len(z_data)):
        freq_array_at_z = np.array(n_c_list_thita1[z_idx])
        flux_array_at_z = np.array(F_obs_1_z[z_idx])
        
        # Interpolate to find the exact flux at our target frequency!
        interpolated_flux = np.interp(target_freq, freq_array_at_z, flux_array_at_z)
        F_obs_1_z_vs_z.append(interpolated_flux)
        
    plt.loglog(z_data, F_obs_1_z_vs_z, marker='o', markersize=3, alpha=0.7, linewidth=2, 
               label=f'$\\nu =$ {target_freq:.0e} Hz', color=color)

plt.xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
plt.ylabel(r'Flux received $F_{\nu, obs}$ (mJy)', fontsize=12)
plt.title('Flux Evolution Along the Jet - Theta=10° (Constant Frequencies)', fontsize=14)
plt.xlim(1, 300)
plt.text(0.98, 0.02, 'Model B  - Observer Frame', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.show()

# ==============================================================================    
# EXTRA PLOT: F_obs_2_z vs Z for different frequencies - THETA 2
# ==============================================================================
plt.figure(figsize=(12, 7))
model_b_name = "Model B:"
z_data = np.array(results[model_b_name]['z']) / Z_0
n_c_list_thita2 = np.array(results[model_b_name]['n_c_list_thita2'])
F_obs_2_z = np.array(results[model_b_name]['F_obs_2_z'])

# Define specific TARGET frequencies to track like a real telescope
target_frequencies = [1e7,1e8, 1e9, 1e10,1e11, 1e12,1e13, 1e14,1e15, 1e16]
cmap = plt.cm.plasma

for j, target_freq in enumerate(target_frequencies):
    color = cmap(j / max(len(target_frequencies) - 1, 1))
    F_obs_2_z_vs_z = []
    
    for z_idx in range(len(z_data)):
        freq_array_at_z = np.array(n_c_list_thita2[z_idx])
        flux_array_at_z = np.array(F_obs_2_z[z_idx])
        
        # Interpolate to find the exact flux at our target frequency!
        interpolated_flux = np.interp(target_freq, freq_array_at_z, flux_array_at_z)
        F_obs_2_z_vs_z.append(interpolated_flux)
        
    plt.loglog(z_data, F_obs_2_z_vs_z, marker='o', markersize=3, alpha=0.7, linewidth=2, 
               label=f'$\\nu =$ {target_freq:.0e} Hz', color=color)

plt.xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
plt.ylabel(r'Flux received $F_{\nu, obs}$ (mJy)', fontsize=12)
plt.title('Flux Evolution Along the Jet - Theta=45° (Constant Frequencies)', fontsize=14)
plt.xlim(1, 300)
plt.text(0.98, 0.02, 'Model B  - Observer Frame', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.show()

# ==============================================================================    
# EXTRA PLOT: Bolometric Flux F vs Z - BOTH THETA ANGLES
# ==============================================================================
plt.figure(figsize=(12, 7))
model_b_name = "Model B:"
z_data = np.array(results[model_b_name]['z']) / Z_0
F_bol_obs_1 = np.array(results[model_b_name]['F_bol_obs_1'])
F_bol_obs_2 = np.array(results[model_b_name]['F_bol_obs_2'])

plt.loglog(z_data, F_bol_obs_1, marker='o', markersize=4, alpha=0.8, linewidth=2.5, 
           label=r'$\theta = 10°$', color='#e74c3c')
plt.loglog(z_data, F_bol_obs_2, marker='s', markersize=4, alpha=0.8, linewidth=2.5, 
           label=r'$\theta = 45°$', color='#3498db')

plt.xlabel(r'Distance $z$ ($Z_0$)', fontsize=12)
plt.ylabel(r'Bolometric Flux $\Sigma(\nu F_\nu)$ (mJy)', fontsize=12)
plt.title('Bolometric Flux vs Distance (Both Viewing Angles)', fontsize=14)
plt.xlim(1, 300)
plt.text(0.98, 0.02, r'Model B  - Observer Frame ($\nu \geq 10^7$ Hz)', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=12, loc='best')
plt.tight_layout()
plt.show()

# ==============================================================================
# 6. RADIATIVE EFFICIENCY CHECK (L_lab vs L_jet) - SANITY CHECK PLOT
# ==============================================================================

print("\n" + "="*70)
print("RADIATIVE EFFICIENCY CHECK (Synchrotron Luminosity vs Jet Power)")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

for idx, model in enumerate(models):
    name = model['name']
    ax = axes[idx]
    
    if 'L_lab' in results[name] and len(results[name]['L_lab']) > 0:
        z_data = np.array(results[name]['z']) / Z_0
        L_lab_array = np.array(results[name]['L_lab'])
        valid_mask = np.isfinite(L_lab_array)
        
        l_total_array = np.array(results[name]['l_total'])
        Total_Jet_Power = l_total_array[np.isfinite(l_total_array)][0]
        
        ax.loglog(z_data[valid_mask], L_lab_array[valid_mask], marker='o', markersize=4, 
                  linewidth=2, label='Synchrotron Power (L_lab)', color=model['color'], alpha=0.8)
        
        ax.axhline(Total_Jet_Power, color='blue', linestyle='-', linewidth=2.5, 
                   label=f'Total Jet Power = {Total_Jet_Power:.2e} erg/s')
        
        threshold_1percent = 0.01 * Total_Jet_Power
        ax.axhline(threshold_1percent, color='red', linestyle='--', linewidth=2, 
                   label=f'1% Threshold = {threshold_1percent:.2e} erg/s')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 300)
        ax.set_xlabel(r'Distance $z$ ($Z_0$)', fontsize=11)
        ax.set_ylabel(r'Luminosity (erg/s)', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        
        max_lum = np.max(L_lab_array[valid_mask])
        min_lum = np.min(L_lab_array[valid_mask])
        max_ratio = max_lum / Total_Jet_Power * 100
        min_ratio = min_lum / Total_Jet_Power * 100
        
        print(f"\n{name}:")
        print(f"  Total Jet Power (L_j):        {Total_Jet_Power:.4e} erg/s")
        print(f"  Max L_lab (at any z):         {max_lum:.4e} erg/s ({max_ratio:.4f}% of L_j)")
        print(f"  Min L_lab (at any z):         {min_lum:.4e} erg/s ({min_ratio:.4f}% of L_j)")
        print(f"  All L_lab values crossing 1% threshold? {np.all(L_lab_array[valid_mask] > threshold_1percent)}")
    else:
        print(f"\n{name}: L_lab data not found. Make sure it is returned in get_properties!")

plt.tight_layout()
plt.show()

print("\n" + "="*70)