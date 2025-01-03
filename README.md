<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">Option Pricing Platform</h3>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#the-pricing-engine">The Pricing Engine</a>
      <ul>
        <li><a href="#historical-volatility">Historical Volatility</a></li>
        <li><a href="#generate-paths">Generate Paths</a></li>
        <li><a href="#calculate-payoffs">Calculate Payoffs</a></li>
        <li><a href="#price-option">Price Option</a></li>
        <li><a href="#finding-greeks">Finding Greeks</a></li>
      </ul>
    </li>
    <li><a href="#next-steps">Next Steps</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![Project Demo](assets/demo.gif)

This is an options pricing platform that I built while trying to learn about derivatives and options. I figured the best way to learn this stuff was to actually build something, so I went ahead and implemented Monte Carlo simulation.

I split it into:
* The pricing engine: Everything related to the pricing logic is here 
* Dashboard UI: Everything related to visualizing the simulation is here

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Streamlit][Streamlit.io]][Streamlit-url]
* [![yFinance][yFinance]][yFinance-url]
* [![Pandas][Pandas]][Pandas-url]
* [![Plotly][Plotly]][Plotly-url]
* [![NumPy][NumPy]][NumPy-url]
* [![Matplotlib][Matplotlib]][Matplotlib-url]
* [![Scipy][Scipy]][Scipy-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## The Pricing Engine

I started with Monte Carlo simulation which is arguably the most famous options pricer. I plan to add Black-Scholes and binomial models to the project later. The simulation works by generating multiple possible price paths that the underlying asset (stocks etc) might take and averaging the payoffs to estimate the option price.

The price movements in each path are generated from a normal distribution (following the standard assumptions of Geometric Brownian Motion). I'm using daily returns with the stock's historical volatility to simulate how the price might change over time. Each path incorporates key factors like:

* The risk-free rate (because future money is worth less than current money)
* Stock volatility (how much the price tends to jump around)
* Time to expiration (longer time means more uncertainty)

By running N paths and averaging the payoffs at expiration (discounted back to present value), I get an estimate of what an option should be worth. 

### Historical Volatility

Measure of degree of variation in the price of an asset over a given time period. This functions does the following:

* Compute daily log return using $r_t = \ln(\frac{P_t}{P_{t-1}})$ where $P_{t}$ is closing price on day $t$
* Compute the standard deviation ($\sigma$) of the return ($r_t$)
* Compute annuliazed returns by multiplying $\sqrt{252}$ (assuming 252 trading days) with $\sigma$

### Generate Paths

The generate_paths function simulates multiple price paths using the Geometric Brownian Motion. This functions works by first 

- Computing the time step $\Delta t = \frac{T}{\text{nsteps}}$ where n_steps is number of time steps
- For each time step $t$ from 1 to n_steps:
  - Define a random variable $z$ from a standard normal distribution.
  - Use the formula:
    $$ S_t = S_{t-1} \cdot \exp\left[\left(r - 0.5 \sigma^2\right) \Delta t + \sigma \sqrt{\Delta t} \cdot z\right]
    $$
    - Where:
      - $S_t$: Stock price at time $t$
      - $r$: Risk-free rate
      - $\sigma$: Historical volatility
      - $\Delta t$: Time step size
      - $z$: Random variable
### Calculate Payoffs

- For call options we use $\text{Payoff}_{call} = max(S_t - K, 0)$
- For put options we use $\text{Payoff}_{put} = max(K - S_t, 0)$
  - Where:
    - $K$ is the strike price
 
### Price Option

This functions calls generate_paths and calculate_payoff functions with parameters n_simulations and n_steps. Using the paths and payoffs the function does the following

- Compute discounted payoffs to present value using:
 $$\text{option}_{price} = e^{-rT} \cdot \text{Mean}(\text{payoffs})$$
  - Where:
    - $e^{-rT}$: Discount factor
    - $\text{Mean}(\text{payoffs})$: Average payoffs of all simulations
- Compute the standard error of the simulation using:
  $$
  \text{Standard Error} = \frac{\sigma}{\sqrt{n_{\text{simulations}}}}
  $$
  - $\sigma$: Standard deviation of the payoffs.

### Finding Greeks

#### calculate_delta

Delta is the rate of change of option price with the change in price of the underlying asset. We use finite differences to approximate Delta because finding the derivative is computationally complex. 

$$
\Delta \approx \frac{V(S_0 + h) - V(S_0 - h)}{2hS_0}
$$

Where:
- $V(S_0 + h) $: Option price when the stock price is increased slightly by a factor $h$
- $V(S_0 - h) $: Option price when the stock price is decreased slightly by a factor $h$
- $h$: Small perturbation in the stock price

#### calculate_gamma

Gamma is the derivative of delta. In other words it measures how much delta changes with the change of price of the underlying asset. 

Using the finite differences method, Gamma is approximated as:

$$
\Gamma \approx \frac{V(S_0 + h) - 2V(S_0) + V(S_0 - h)}{(hS_0)^2}
$$

Where:
- $V(S_0)$: Option price at the original stock price.

#### calculate_theta

Theta is the rate of change of option price with respect to time. In simpled terms, theta calculates how much option price decreases as time gets closer to expiration date. 

Using finite differences, Theta is approximated as:

$$
\Theta \approx -\frac{V(T + h) - V(T)}{h}
$$

Where:
- $V(T)$: Current option price.
- $V(T + h)$: Option price when the time to maturity is slightly increased by $h$.
- $h$: Small perturbation in the time to maturity.

#### calculate_vega

Vega is the measure of how the option price changes when the volatility of the underlying asset changes. 

Using the finite differences method, Vega is approximated as:

$$
\nu \approx \frac{V(\sigma + h) - V(\sigma - h)}{2h}
$$

Where:
- $V(\sigma + h)$: Option price when the volatility is increased by a small factor $h$.
- $V(\sigma - h)$: Option price when the volatility is decreased by a small factor $h$.
- $h$: Small perturbation in the volatility.



<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Next Steps

- [ ] Implement Black-Scholes model
- [ ] Implement Binomial model


<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Top contributors:

<a href="https://github.com/Satvikkapoor/OptionPricingAndAnalysis/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Satvikkapoor/OptionPricingAndAnalysis" alt="contrib.rocks image" />
</a>



<!-- CONTACT -->
## Contact

Your Name - [@SatvikKapoor11](https://twitter.com/@SatvikKapoor11) - ksatvik@gmail.com

Project Link: [https://github.com/Satvikkapoor/OptionPricingAndAnalysis](https://github.com/Satvikkapoor/OptionPricingAndAnalysis)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Satvikkapoor/OptionPricingAndAnalysis.svg?style=for-the-badge
[contributors-url]: https://github.com/Satvikkapoor/OptionPricingAndAnalysis/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Satvikkapoor/OptionPricingAndAnalysis.svg?style=for-the-badge
[forks-url]: https://github.com/Satvikkapoor/OptionPricingAndAnalysis/network/members
[stars-shield]: https://img.shields.io/github/stars/Satvikkapoor/OptionPricingAndAnalysis.svg?style=for-the-badge
[stars-url]: https://github.com/Satvikkapoor/OptionPricingAndAnalysis/stargazers
[issues-shield]: https://img.shields.io/github/issues/Satvikkapoor/OptionPricingAndAnalysis.svg?style=for-the-badge
[issues-url]: https://github.com/Satvikkapoor/OptionPricingAndAnalysis/issues
[license-shield]: https://img.shields.io/github/license/Satvikkapoor/OptionPricingAndAnalysis.svg?style=for-the-badge
[license-url]: https://github.com/Satvikkapoor/OptionPricingAndAnalysis/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/satvik-kapoor
[product-demo]: assets/demo.gif
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
[Streamlit.io]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/
[yFinance]: https://img.shields.io/badge/yFinance-1E90FF?style=for-the-badge&logo=yahoo&logoColor=white
[yFinance-url]: https://github.com/ranaroussi/yfinance
[Pandas]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Plotly]: https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white
[Plotly-url]: https://plotly.com/
[NumPy]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Matplotlib]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
[Scipy]: https://img.shields.io/badge/Scipy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white
[Scipy-url]: https://scipy.org/
