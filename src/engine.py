import torch

class DaviesModel:
    def __init__(self, dij, Ii, Zj, device=None, Nt=500, Ntt=10):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu") # FIXED: #2
        self.dij = torch.from_numpy(dij).to(self.device).float()

        self.Ii = torch.from_numpy(Ii).to(self.device).float()
        self.Zj = torch.from_numpy(Zj).to(self.device).float()

        # Constants
        self.nlat = 78
        self.nlon = 71
        self.nz = 500
        self.Lr = 6
        self.Lp = 12
        self.Nt = Nt
        self.Ntt = Ntt
        self.dt = 0.1
        self.eta = 0.005
        self.tau = 0.75
        self.Ptotal = 500.0

    def run_simulation(self, beta_r, gamma_r, alpha_p, gamma_p):
        """
        Runs the Davies model simulation.

        Args:
            beta_r (float): Distance decay parameter for rioters.
            gamma_r (float): Deterrent effect of police presence.
            alpha_p (float): Attractiveness of a location to police.
            gamma_p (float): Impact of the number of rioters on the police requirement.

        Returns:
            np.ndarray: The number of rioters in each zone (Rj) at the end of the simulation.
        """
        # Initialize tensors
        Rj_t = torch.zeros(self.nz, device=self.device, dtype=torch.float32)
        Ai_t = torch.zeros((self.nlat, self.nlon), device=self.device, dtype=torch.float32)
        Pj_t = torch.zeros(self.nz, device=self.device, dtype=torch.float32)
        fjdel_t = torch.zeros((self.nz, self.Lr), device=self.device, dtype=torch.float32)
        rho_t = torch.ones((self.nlat, self.nlon), device=self.device, dtype=torch.float32)
        Ddel_t = torch.zeros((self.nz, self.Lp), device=self.device, dtype=torch.float32)
        Dj_t = torch.zeros(self.nz, device=self.device, dtype=torch.float32)

        # Local copy of Ii to avoid modifying the instance variable
        Ii_t = self.Ii.clone()

        # Precompute dij term
        # Note: self.dij is already on device
        auxij1 = torch.exp(-beta_r * self.dij)
        target_values = self.Zj[:, 2]
        dij_t = 1.0 * target_values * auxij1 / torch.max(target_values)

        counter = 0

        for nn in range(self.Nt):
            for mm in range(self.Ntt):
                # Calculate fj_t (Attractiveness modification by police - Deterrence)
                fj_t = torch.exp(-torch.floor(gamma_r * Pj_t / (Rj_t + 1.0e-20)))

                # Effective Attractiveness
                Wij_t = fj_t * dij_t
                Wi_t = Wij_t.sum(dim=2)
                
                # Probability of rioting
                P_off_t = rho_t * Wi_t / (1.0 + Wi_t)

                # Delayed attractiveness effect storage (fjdel)
                idxr = counter % self.Lr
                fjdel_t[:, idxr] = fj_t
                dnm_r = self.Lr if counter >= self.Lr else counter + 1
                We_ij_t = fjdel_t.sum(dim=1) * dij_t / dnm_r

                # Flow computation
                # Step 1: Normalization factor
                auxw = Ai_t / (We_ij_t.sum(dim=2) + 1.0e-20)
                
                # Step 2: Spatial distribution
                Sij_t = auxw.unsqueeze(2) * We_ij_t

                # Rioter population at sites (Rj)
                Rj_t = torch.sum(Sij_t.sum(dim=1), dim=0)

                # Police Requirement (Dj)
                Dj_t[:] = target_values ** (alpha_p) * torch.exp(gamma_p * Rj_t[:])

                # Delayed police requirement storage (Ddel)
                idxp = counter % self.Lp
                Ddel_t[:, idxp] = Dj_t[:]
                dnm_p = self.Lp if counter >= self.Lp else counter + 1
                Dej_t = torch.sum(Ddel_t, dim=1) / dnm_p

                # Police allocation (Pj)
                Pj_t = self.Ptotal * Dej_t / (Dej_t.sum() + 1.0e-20)

                # Increment counter ONCE per sub-step
                counter += 1

                # Capture Rate (fj_cap) based on updated police
                fj_cap = 1.0 - torch.exp(-torch.floor(Pj_t / (Rj_t + 1.0e-20)))
                Ci_t = self.tau * torch.sum(Sij_t * fj_cap, dim=2)

                # Evolution of Active (Ai) and Inactive (Ii) populations
                Ai_t += self.dt * (self.eta * P_off_t * Ii_t - Ci_t)
                Ii_t += -self.dt * self.eta * P_off_t * Ii_t

        return Rj_t.cpu().numpy()
