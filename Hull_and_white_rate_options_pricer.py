import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import datetime
import pandas_datareader.data as web
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Retrieve rate data for the underlying.
def get_rate_data(underlying):
    if underlying == "US LIBOR":
        try:
            start = datetime.datetime.now() - datetime.timedelta(days=30)
            end = datetime.datetime.now()
            data = web.DataReader("USDONTD156N", "fred", start, end)
            current_rate = data.iloc[-1, 0] / 100.0
            disclaimer = f"Data as of {data.index[-1].strftime('%Y-%m-%d')}"
            return current_rate, disclaimer
        except Exception as e:
            return 0.025, "Data retrieval failed, using default rate"
    elif underlying == "IOS curve":
        return 0.03, "Using dummy IOS curve rate"
    else:
        return 0.025, "Default rate used"

# Build a simplified binomial tree for Hull–White dynamics.
def build_rate_tree(r0, a, sigma, T, N):
    dt = T / N
    dx = sigma * np.sqrt(dt)
    tree = []
    for i in range(N + 1):
        level = []
        for j in range(i + 1):
            rate = r0 + (2 * j - i) * dx
            level.append(rate)
        tree.append(level)
    return tree, dt, dx

# Price the option using backward induction on the tree.
def price_option(r0, a, sigma, T, N, option_type, strike, barrier_type, barrier, is_barrier, is_bermudan, is_call):
    tree, dt, dx = build_rate_tree(r0, a, sigma, T, N)
    option_values = np.zeros((N + 1, N + 1))
    # Terminal payoff.
    for j in range(N + 1):
        r = tree[N][j]
        intrinsic = (r - strike) if is_call else (strike - r)
        option_values[N][j] = max(intrinsic, 0)
        if is_barrier and barrier_type is not None:
            if barrier_type in ["Down In", "Up In"]:
                if not (r <= barrier if barrier_type=="Down In" else r >= barrier):
                    option_values[N][j] = 0
            elif barrier_type in ["Down Out", "Up Out"]:
                if (r <= barrier if barrier_type=="Down Out" else r >= barrier):
                    option_values[N][j] = 0

    p = 0.5  # risk-neutral probability approximation
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            disc = np.exp(-tree[i][j] * dt)
            continuation = disc * (p * option_values[i + 1][j + 1] + (1 - p) * option_values[i + 1][j])
            if is_bermudan:
                intrinsic = (tree[i][j] - strike) if is_call else (strike - tree[i][j])
                intrinsic = max(intrinsic, 0)
                option_values[i][j] = max(continuation, intrinsic)
            else:
                option_values[i][j] = continuation
            if is_barrier and barrier_type is not None:
                if barrier_type in ["Down In", "Up In"]:
                    if not (tree[i][j] <= barrier if barrier_type=="Down In" else tree[i][j] >= barrier):
                        option_values[i][j] = 0
                elif barrier_type in ["Down Out", "Up Out"]:
                    if (tree[i][j] <= barrier if barrier_type=="Down Out" else tree[i][j] >= barrier):
                        option_values[i][j] = 0
    return option_values[0][0]

# A custom widget linking a slider and an entry.
class LabeledSlider(ttk.Frame):
    def __init__(self, master, label, from_, to, initial, resolution=0.001, **kwargs):
        super().__init__(master, **kwargs)
        self.var = tk.DoubleVar(value=initial)
        self.resolution = resolution

        self.label = ttk.Label(self, text=label)
        self.entry = ttk.Entry(self, textvariable=self.var, width=7)
        self.slider = ttk.Scale(self, orient="horizontal", from_=from_, to=to, variable=self.var, command=self.slider_updated)
        
        self.label.grid(row=0, column=0, padx=5)
        self.entry.grid(row=0, column=1, padx=5)
        self.slider.grid(row=0, column=2, padx=5, sticky="ew")
        self.columnconfigure(2, weight=1)
        
        # Trace variable changes to update entry live.
        self.var.trace_add("write", self.entry_updated)
        self.entry.bind("<Return>", self.manual_entry_update)

    def slider_updated(self, event):
        try:
            value = self.var.get()
            self.entry.delete(0, tk.END)
            self.entry.insert(0, f"{value:.4f}")
        except Exception:
            pass

    def entry_updated(self, *args):
        try:
            value = self.var.get()
            self.entry.delete(0, tk.END)
            self.entry.insert(0, f"{value:.4f}")
        except Exception:
            pass

    def manual_entry_update(self, event):
        try:
            val = float(self.entry.get())
            self.var.set(val)
        except ValueError:
            pass

# Main Application with shared parameters and a Notebook with one tab.
class HullWhiteApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        master.title("Hull-White Option Pricing App")
        master.geometry("950x800")
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.create_widgets()

    def create_widgets(self):
        # Shared parameters frame.
        self.param_frame = ttk.LabelFrame(self, text="Option Parameters", padding=10)
        self.param_frame.pack(fill="x", padx=10, pady=10)

        row = 0
        ttk.Label(self.param_frame, text="Underlying:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.underlying_var = tk.StringVar(value="US LIBOR")
        underlying_options = ["US LIBOR", "IOS curve"]
        ttk.OptionMenu(self.param_frame, self.underlying_var, self.underlying_var.get(), *underlying_options).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        row += 1
        ttk.Label(self.param_frame, text="Option Type:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.option_type_var = tk.StringVar(value="European")
        option_types = ["European", "Bermudan"]
        ttk.OptionMenu(self.param_frame, self.option_type_var, self.option_type_var.get(), *option_types).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        row += 1
        ttk.Label(self.param_frame, text="Barrier ?").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.barrier_choice_var = tk.StringVar(value="No")
        barrier_choices = ["Yes", "No"]
        barrier_menu = ttk.OptionMenu(self.param_frame, self.barrier_choice_var, self.barrier_choice_var.get(), *barrier_choices, command=self.toggle_barrier)
        barrier_menu.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        row += 1
        self.barrier_type_frame = ttk.Frame(self.param_frame)
        self.barrier_type_frame.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Label(self.barrier_type_frame, text="Barrier Type:").grid(row=0, column=0, sticky="w", padx=5)
        self.barrier_type_var = tk.StringVar(value="Down In")
        barrier_options = ["Down In", "Down Out", "Up In", "Up Out"]
        self.barrier_type_menu = ttk.OptionMenu(self.barrier_type_frame, self.barrier_type_var, self.barrier_type_var.get(), *barrier_options)
        self.barrier_type_menu.grid(row=0, column=1, sticky="w", padx=5)
        self.barrier_type_frame.grid_remove()  # hide initially
        
        # Sliders for numeric parameters.
        row += 1
        self.strike_slider = LabeledSlider(self.param_frame, "Strike:", 0.00, 0.10, 0.02)
        self.strike_slider.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        row += 1
        self.barrier_slider = LabeledSlider(self.param_frame, "Barrier Level:", 0.00, 0.10, 0.015)
        self.barrier_slider.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        row += 1
        self.maturity_slider = LabeledSlider(self.param_frame, "Maturity (years):", 0.5, 10, 5, resolution=0.1)
        self.maturity_slider.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        row += 1
        self.steps_slider = LabeledSlider(self.param_frame, "Time Steps:", 10, 200, 50, resolution=1)
        self.steps_slider.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        row += 1
        self.a_slider = LabeledSlider(self.param_frame, "Mean Reversion (a):", 0.01, 1.0, 0.1)
        self.a_slider.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        row += 1
        self.sigma_slider = LabeledSlider(self.param_frame, "Volatility (sigma):", 0.001, 0.2, 0.01)
        self.sigma_slider.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        row += 1
        ttk.Label(self.param_frame, text="Option - Call/Put:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.call_put_var = tk.StringVar(value="Call")
        ttk.OptionMenu(self.param_frame, self.call_put_var, self.call_put_var.get(), "Call", "Put").grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        row += 1
        self.disclaimer_label = ttk.Label(self.param_frame, text="Disclaimer: No data loaded yet.")
        self.disclaimer_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Notebook with one tab: Option Price.
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.price_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.price_tab, text="Option Price")
        self.price_button = ttk.Button(self.price_tab, text="Price Option", command=self.start_price_thread)
        self.price_button.pack(pady=5)
        self.price_result = ttk.Label(self.price_tab, text="Option Price: ", font=("Helvetica", 24))
        self.price_result.pack(pady=10)
        
        self.pack(fill="both", expand=True)

    def toggle_barrier(self, value):
        if value == "Yes":
            self.barrier_type_frame.grid()
        else:
            self.barrier_type_frame.grid_remove()

    def start_price_thread(self):
        thread = threading.Thread(target=self.compute_price)
        thread.start()

    def compute_price(self):
        try:
            underlying = self.underlying_var.get()
            option_type = self.option_type_var.get()
            barrier_choice = self.barrier_choice_var.get()
            barrier_type = self.barrier_type_var.get() if barrier_choice == "Yes" else None
            strike = float(self.strike_slider.var.get())
            barrier_level = float(self.barrier_slider.var.get())
            T = float(self.maturity_slider.var.get())
            N = int(float(self.steps_slider.var.get()))
            a = float(self.a_slider.var.get())
            sigma = float(self.sigma_slider.var.get())
            is_call = (self.call_put_var.get() == "Call")
            is_bermudan = (option_type == "Bermudan")
            is_barrier = (barrier_choice == "Yes")
            
            r0, disclaimer = get_rate_data(underlying)
            self.disclaimer_label.after(0, lambda: self.disclaimer_label.config(text=f"Disclaimer: {disclaimer}"))
            price = price_option(r0, a, sigma, T, N, option_type, strike, barrier_type, barrier_level, is_barrier, is_bermudan, is_call)
            self.price_result.after(0, lambda: self.price_result.config(text=f"Option Price: {price:.6f}"))
        except Exception as e:
            self.price_result.after(0, lambda: messagebox.showerror("Error", f"An error occured: {e}"))

if __name__ == "__main__":
    root = tk.Tk()
    app = HullWhiteApp(root)
    root.mainloop()
