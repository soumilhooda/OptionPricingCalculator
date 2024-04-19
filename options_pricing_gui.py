import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from options_pricing_functions import *  
class OptionPricingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Option Pricing Calculator")

        # Create Notebook (tabbed interface)
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Page 1: Models and Results ---
        models_frame = tk.Frame(self.notebook)
        self.notebook.add(models_frame, text="Models & Results")

        # Input fields and labels
        self.input_fields = {}
        input_labels = [
            "Initial Stock Price (S0):",
            "Strike Price (K):",
            "Time to Maturity (T):",
            "Risk-Free Rate (r):",
            "Volatility (sigma):",
            "Dividend Rate (d):"
        ]

        for i, label in enumerate(input_labels):
            label_widget = tk.Label(models_frame, text=label)
            label_widget.grid(row=i, column=0, padx=5, pady=2)
            entry_widget = tk.Entry(models_frame)
            entry_widget.grid(row=i, column=1, padx=5, pady=2)
            self.input_fields[label] = entry_widget

        # Option type and model selection
        self.option_type_var = tk.StringVar(models_frame)
        self.option_type_var.set("Call")
        option_type_options = ["Call", "Put"]
        option_type_menu = tk.OptionMenu(models_frame, self.option_type_var, *option_type_options)
        option_type_menu.grid(row=6, column=0, columnspan=2, pady=5)

        self.model_var = tk.StringVar(models_frame)
        self.model_var.set("CRR")
        model_options = ["CRR", "Jarrow-Rudd", "Tian", "Trinomial", "Monte Carlo", "MOU"]
        model_menu = tk.OptionMenu(models_frame, self.model_var, *model_options)
        model_menu.grid(row=7, column=0, columnspan=2, pady=5)

        # Calculate button and result label
        self.calculate_button = tk.Button(models_frame, text="Calculate", command=self.calculate_and_display)
        self.calculate_button.grid(row=8, column=0, columnspan=2, pady=10)

        self.result_label = tk.Label(models_frame, text="")
        self.result_label.grid(row=9, column=0, columnspan=2)

        # Greeks
        greeks_labels = ["Delta:", "Gamma:", "Vega:", "Theta:", "Rho:"]
        self.greeks_values = {}

        for i, label in enumerate(greeks_labels):
            label_widget = tk.Label(models_frame, text=label)
            label_widget.grid(row=10 + i, column=0, padx=5, pady=2)
            value_label = tk.Label(models_frame, text="")
            value_label.grid(row=10 + i, column=1, padx=5, pady=2)
            self.greeks_values[label] = value_label

        # --- Page 2: Plotting ---
        plot_frame = tk.Frame(self.notebook)
        self.notebook.add(plot_frame, text="Plotting")

        # Plot canvas and button
        self.plot_canvas = tk.Canvas(plot_frame, width=400, height=300, bg="white")
        self.plot_canvas.grid(row=0, column=0, columnspan=2)

        self.plot_button = tk.Button(plot_frame, text="Generate Plot", command=self.generate_plot)
        self.plot_button.grid(row=1, column=0, columnspan=2, pady=10)

        # Min/Max Strike input
        self.min_strike_entry = tk.Entry(plot_frame)
        self.min_strike_entry.grid(row=2, column=1, padx=5, pady=5)
        tk.Label(plot_frame, text="Min Strike:").grid(row=2, column=0, padx=5, pady=5)

        self.max_strike_entry = tk.Entry(plot_frame)
        self.max_strike_entry.grid(row=3, column=1, padx=5, pady=5)
        tk.Label(plot_frame, text="Max Strike:").grid(row=3, column=0, padx=5, pady=5)

        # --- Page 3: Risk Reversal ---
        rr_frame = tk.Frame(self.notebook)
        self.notebook.add(rr_frame, text="Risk Reversal")

        # Risk Reversal input fields
        self.k1_entry = tk.Entry(rr_frame)
        self.k1_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(rr_frame, text="K1 (Put Strike):").grid(row=0, column=0, padx=5, pady=5)

        self.k2_entry = tk.Entry(rr_frame)
        self.k2_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Label(rr_frame, text="K2 (Call Strike):").grid(row=1, column=0, padx=5, pady=5)

        # Dividend Rate input
        self.dividend_rate_entry = tk.Entry(rr_frame)
        self.dividend_rate_entry.grid(row=2, column=1, padx=5, pady=5)
        tk.Label(rr_frame, text="Dividend Rate (d):").grid(row=2, column=0, padx=5, pady=5)

        # Risk Reversal result label
        self.rr_result_label = tk.Label(rr_frame, text="")
        self.rr_result_label.grid(row=2, column=0, columnspan=2, pady=5)

    def calculate_and_display(self):
        try:
            # Get input values
            S0 = float(self.input_fields["Initial Stock Price (S0):"].get())
            K = float(self.input_fields["Strike Price (K):"].get())
            T = float(self.input_fields["Time to Maturity (T):"].get())
            r = float(self.input_fields["Risk-Free Rate (r):"].get())
            sigma = float(self.input_fields["Volatility (sigma):"].get())
            otype = self.option_type_var.get().lower()
            model = self.model_var.get().lower()

            # Call appropriate option pricing function
            if model == "crr":
                option_value = CRR_option_value(S0, K, T, r, sigma, otype)
            elif model == "jarrow-rudd":
                option_value = Jarrow_Rudd_option_value(S0, K, T, r, sigma, otype)
            elif model == "tian":
                option_value = Tian_option_value(S0, K, T, r, sigma, otype)
            elif model == "trinomial":
                option_value = trinomial_option_value(S0, K, T, r, sigma, otype)
            elif model == "monte carlo":
                option_value = monte_carlo_option_value(S0, K, T, r, sigma, otype)
            elif model == "mou":
                # Get dividend rate for MOU
                d = float(self.input_fields["Dividend Rate (d):"].get())
                option_value = MOU_option_value(S0, K, T, r, sigma, otype, kappa=5, theta=100) 
            else:
                raise ValueError("Invalid model selected.")
            self.result_label.config(text=f"Option Value: {option_value:.4f}")
        except ValueError as e:
            self.result_label.config(text=f"Error: {str(e)}")

        # Calculate and display Greeks
        try:
            greeks = calculate_greeks(S0, K, T, r, sigma, otype)
            for i, label in enumerate(self.greeks_values):
                self.greeks_values[label].config(text=f"{greeks[i]:.4f}")
        except ValueError as e:
            self.result_label.config(text=f"Error: {str(e)}")

        # Calculate and display risk reversal
        try:
            k1 = float(self.k1_entry.get())
            k2 = float(self.k2_entry.get())

            # Get dividend rate
            d = float(self.dividend_rate_entry.get())  

            # Get other parameters from input fields (assuming they are available)
            S0 = float(self.input_fields["Initial Stock Price (S0):"].get())
            T = float(self.input_fields["Time to Maturity (T):"].get())
            r = float(self.input_fields["Risk-Free Rate (r):"].get())
            sigma = float(self.input_fields["Volatility (sigma):"].get())


            # Create RiskReversal object and calculate PV
            rr = RiskReversal(S0, k1, k2, T, r, sigma, d)
            pv = rr.calculate_pv()

            self.rr_result_label.config(text=f"Risk Reversal PV: {pv:.4f}")
        except ValueError as e:
            self.rr_result_label.config(text=f"Error: {str(e)}")

    def generate_plot(self):
        try:
            S0 = float(self.input_fields["Initial Stock Price (S0):"].get())
            T = float(self.input_fields["Time to Maturity (T):"].get())
            r = float(self.input_fields["Risk-Free Rate (r):"].get())
            sigma = float(self.input_fields["Volatility (sigma):"].get())
            otype = self.option_type_var.get().lower()
            model = self.model_var.get().lower()

            # Get strike price range from user input (add input fields for min/max strike)
            min_strike = float(self.min_strike_entry.get())
            max_strike = float(self.max_strike_entry.get())
            strikes = range(int(min_strike), int(max_strike) + 1)
            # Calculate option prices based on selected model
            option_prices = []
            for K in strikes:
                if model == "bsm":
                    option_prices.append(BSM_option_value(S0, K, T, r, sigma, otype))
                elif model == "crr":
                    option_prices.append(CRR_option_value(S0, K, T, r, sigma, otype))
                elif model == "jarrow-rudd":
                    option_prices.append(Jarrow_Rudd_option_value(S0, K, T, r, sigma, otype))
                elif model == "tian":
                    option_prices.append(Tian_option_value(S0, K, T, r, sigma, otype))
                elif model == "trinomial":
                    option_prices.append(trinomial_option_value(S0, K, T, r, sigma, otype))
                elif model == "monte carlo":
                    option_prices.append(monte_carlo_option_value(S0, K, T, r, sigma, otype))
                elif model == "mou":
                    d = float(self.input_fields["Dividend Rate (d):"].get())
                    option_prices.append(MOU_option_value(S0, K, T, r, sigma, otype, kappa=5, theta=100))
                else:
                    raise ValueError("Invalid model selected.")
            # Create plot using Matplotlib
            fig, ax = plt.subplots()
            ax.plot(strikes, option_prices, label=otype.title())
            ax.set_xlabel("Strike Price")
            ax.set_ylabel("Option Price")
            ax.set_title(f"{model.upper()} Option Pricing")
            ax.legend()

            # Embed plot in canvas
            canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas.draw()  
        except ValueError as e:
            self.result_label.config(text=f"Error: {str(e)}")

# Create the GUI
root = tk.Tk()
gui = OptionPricingGUI(root)

# Center the GUI contents
root.update_idletasks()  # Required to get accurate width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (root.winfo_width() / 2))
y = int((screen_height / 2) - (root.winfo_height() / 2))
root.geometry(f"+{x}+{y}")

root.mainloop()