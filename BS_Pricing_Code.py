import numpy as np
from scipy.stats import norm
from datetime import datetime, date
import matplotlib.pyplot as plt 

def black_scholes_price(S0, K, r, sigma, T, option_type='call'):
    """ Cela calcule le prix d'une option européenne avec le modèle de Black-Scholes.
    
    Paramètres :
    - S0 : Prix actuel du sous-jacent (> 0).
    - K : Prix d'exercice (strike price, > 0).
    - r : Taux sans risque (annualisé, >= 0).
    - sigma : Volatilité annuelle (> 0).
    - T : Temps jusqu'à l'échéance (années, > 0).
    - option_type : Type d'option ('call' ou 'put').
    
    Cela retourne le prix de l'option."""
    try:
        if S0 <= 0:
            raise ValueError("Le prix actuel du sous-jacent (S0) doit être > 0.")
        if K <= 0:
            raise ValueError("Le prix d'exercice (K) doit être > 0.")
        if r < 0:
            raise ValueError("Le taux sans risque (r) doit être >= 0.")
        if sigma <= 0:
            raise ValueError("La volatilité (sigma) doit être > 0.")
        if T <= 0:
            raise ValueError("Le temps jusqu'à l'échéance (T) doit être > 0.")
        if option_type not in ['call', 'put']:
            raise ValueError("Le type d'option doit être 'call' ou 'put'.")
        
        # Calcul de d1 et d2
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Prix de l'option
        if option_type == 'call':
            price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        
        return price
    
    except ValueError as erreur:
        print(f"Erreur : {erreur}")
        return None


def calculate_days_remaining(purchase_date, expiration_date):
    """ Cela calcule le nombre de jours restants entre la date d'achat et la date d'expiration.
    
    Paramètres :
    - purchase_date : Date d'achat (format YYYY-MM-DD).
    - expiration_date : Date d'expiration (format YYYY-MM-DD).
    
    Cela retourne le temps restant en jours."""
    try:
        # On convertit les dates en objets datetime pour les manipuler
        purchase_date = datetime.strptime(purchase_date, "%Y-%m-%d").date()
        expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
        
        # On vérifie que les dates sont valides
        if purchase_date > expiration_date:
            raise ValueError("La date d'achat ne peut pas être après la date d'expiration.")
        
        # On calcule le nombre de jours restants
        days_remaining = (expiration_date - today).days
        return days_remaining
    
    except ValueError as erreur:
        print(f"Erreur : {erreur}")
        return None


def plot_payoff(K, premium, option_type):
    """Cela trace le payoff d'une option spécifique (Call ou Put) en fonction du prix du sous-jacent à l'échéance.
    
    Paramètres :
    - K : Prix d'exercice (strike price).
    - premium : Prime payée pour l'option.
    - option_type : Type d'option ('call' ou 'put')."""
    # Plage de prix du sous-jacent
    S = np.linspace(0.5 * K, 1.5 * K, 500)
    
    # Calcul du payoff en fonction du type d'option
    if option_type == 'call':
        payoff = np.maximum(S - K, 0) - premium
        label = "Payoff Call"
        color = "blue"
    elif option_type == 'put':
        payoff = np.maximum(K - S, 0) - premium
        label = "Payoff Put"
        color = "red"
    else:
        raise ValueError("Le type d'option doit être 'call' ou 'put'.")
    
    # Tracer le payoff
    plt.figure(figsize=(10, 6))
    plt.plot(S, payoff, label=label, color=color)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Prix du Sous-Jacent à l'Échéance")
    plt.ylabel("Payoff")
    plt.title(f"Payoff de l'Option {option_type.upper()}")
    plt.legend()
    plt.grid()
    plt.show()


# Point d'entrée du programme qui s'execute automatiquement si on ouvre le fichier
if __name__ == "__main__":
    print("Calcul du Prix d'une Option Européenne avec Black-Scholes")
    try:
        # Entrées utilisateur
        S0 = float(input("Entrez le prix actuel du sous-jacent (S0, > 0) : "))
        K = float(input("Entrez le prix d'exercice (K, > 0) : "))
        r = float(input("Entrez le taux sans risque (r, ex. 0.05 pour 5%, >= 0) : "))
        sigma = float(input("Entrez la volatilité (sigma, ex. 0.2 pour 20%, > 0) : "))
        purchase_date = input("Entrez la date d'achat (format YYYY-MM-DD) : ")
        expiration_date = input("Entrez la date d'expiration (format YYYY-MM-DD) : ")
        option_type = input("Entrez le type d'option ('call' ou 'put') : ").strip().lower()
        
        # Calcul du nombre de jours restants
        days_remaining = calculate_days_remaining(purchase_date, expiration_date)
        if days_remaining is None:
            exit()
        
        # Conversion des jours restants en années
        T = days_remaining / 365
        
        # Afficher le nombre de jours restants
        print(f"\nNombre de jours restants jusqu'à l'échéance : {days_remaining} jours.")
        
        # Calcul du prix de l'option
        price = black_scholes_price(S0, K, r, sigma, T, option_type)
        if price is not None:
            print(f"Prix de l'option {option_type.upper()} : {price:.2f}")
            
            # Tracer les payoffs si demandé
            if input("Voulez-vous tracer les payoffs ? (o/n) : ").strip().lower() == 'o':
                premium = price  # La prime est égale au prix calculé
                plot_payoff(K, premium, option_type)
    
    except ValueError as erreur:
        print(f"Erreur : {erreur}")
