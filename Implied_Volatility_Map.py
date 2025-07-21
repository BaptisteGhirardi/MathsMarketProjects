import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, date
from scipy.stats import norm
from matplotlib.ticker import ScalarFormatter
from BS_Pricing_Code import black_scholes_price, calculate_days_remaining

def vega(S0, K, r, sigma, T):
    """Calcule le Vega (sensibilité du prix à la volatilité). On utilise pour la méthode Newton-Raphson."""

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * np.sqrt(T) * norm.pdf(d1)

# Calcul de la volatilité implicite
def implied_volatility_newton(S0, K, r, T, market_price, option_type='call', max_iter=300, tol=1e-6, initial_guess=0.3):
    sigma = initial_guess

    for _ in range(max_iter):
        price = black_scholes_price(S0, K, r, sigma, T, option_type)
        if price is None:
            return None
        
        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        v = vega(S0, K, r, sigma, T)
        if v == 0:
            return sigma

        sigma -= diff / v

    print("Avertissement : Pas de convergence après", max_iter, "itérations.")
    return sigma


# Point d'entrée du programme
if __name__ == "__main__":
    print("Nappe de Volatilité Implicite")
    
    try:
        # Paramètres utilisateur
        S0 = float(input("Prix actuel du sous-jacent (S0) : "))
        K_central = float(input("Strike central (K_central) : "))
        r = float(input("Taux sans risque (r) : "))
        option_type = input("Type d'option ('call' ou 'put') : ").strip().lower()

        purchase_date_str = input("Date d'achat (YYYY-MM-DD) : ")
        expiration_date_str = input("Date d'expiration (YYYY-MM-DD) : ")

        days_remaining = calculate_days_remaining(purchase_date_str, expiration_date_str)
        if days_remaining is None:
            exit()
        
        T_central = days_remaining / 365
        print(f"Temps jusqu'à échéance (T) : {T_central:.4f} années")

        # Grille de strikes
        strike_min = K_central - 20
        strike_max = K_central + 20
        strikes = np.linspace(strike_min, strike_max, 50)

        # Grille de maturités 
        maturity_min = 0.1 * T_central
        maturity_max = T_central
        maturities = np.arange(maturity_min, maturity_max + 0.05, 0.05)

        # Tableau 2D pour stocker les résultats
        vol_surface = np.zeros((len(strikes), len(maturities)))

        # Simule des écarts au prix BS pour faire apparaître le smile
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                # Prix BS standard avec sigma=0.2 depuis mon code du fichier BS_Pricing_Code
                market_price = black_scholes_price(S0, K, r, sigma=0.2, T=T, option_type=option_type)
                
                # Simule un léger biais sur le prix pour les options OTM pour rendre le smile plus présent
                if K < S0:  # Put OTM – légère augmentation de prix → IV augmente
                    market_price *= 1.005
                elif K > S0:  # Call OTM – légère augmentation aussi
                    market_price *= 1.003
                    
                # Calcul de la volatilité implicite
                vol = implied_volatility_newton(S0, K, r, T, market_price, option_type)
                vol_surface[i, j] = vol

        # Tracé 3D de la surface de volatilité implicite
        X, Y = np.meshgrid(maturities, strikes)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, vol_surface, cmap='viridis', edgecolor='none', rstride=1, cstride=1)
        ax.set_xlabel('Maturité (années)')
        ax.set_ylabel('Strike Price')
        ax.set_zlabel('Volatilité Implicite (σ)')
        ax.set_title('Nappe de Volatilité Implicite', pad=40)
        plt.show()

    except Exception as e:
        print(f"Erreur : {e}")
