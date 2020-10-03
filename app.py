import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import datetime
import lmfit
from scipy.integrate import odeint
from scipy.special import logit
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def get_death_data(df):
    df = df[~df["FECHA_FALLECIMIENTO"].isnull()]
    try:
        df["FECHA_FALLECIMIENTO"] = df["FECHA_FALLECIMIENTO"].apply(
            lambda x: convert_datetime(x))
    except:
        df["FECHA_FALLECIMIENTO"] = df["FECHA_FALLECIMIENTO"].apply(
            lambda x: convert_datetime_2(x))
    df = df.groupby(by=["FECHA_FALLECIMIENTO"]).size().reset_index()
    df.columns = ["date", "count"]
    df = count_acum(df)
    return df


def count_acum(df):
    df["count_acum"] = 0
    for index, item in df.iterrows():
        if index == 0:
            df.loc[index, "count_acum"] = df.loc[index, "count"]
        else:
            df.loc[index, "count_acum"] = df.loc[index -
                                                 1, "count_acum"] + df.loc[index, "count"]
    return df


def convert_datetime(x):
    date = datetime.datetime.strptime(x, '%d/%m/%Y')
    return date


def convert_datetime_2(x):
    x_date = str(x)[6:] + "/" + str(x)[4:6] + "/" + str(x)[:4]
    date = datetime.datetime.strptime(x_date, '%d/%m/%Y')
    return date


def logistic_R_0(t, R_0_start, k, x0, R_0_end):
    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end


def Model(days, agegroups, beds_per_100k, R_0_start, k, x0, R_0_end, prob_I_to_C, prob_C_to_D, s, gamma, sigma):
    def beta(t):
        return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma

    N = sum(agegroups)

    def Beds(t):
        beds_0 = beds_per_100k / 100_000 * N
        return beds_0 + s*beds_0*t  # 0.003

    y0 = N-1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    t = np.linspace(0, days-1, days)
    ret = odeint(deriv, y0, t, args=(beta, gamma, sigma,
                                     N, prob_I_to_C, prob_C_to_D, Beds))
    S, E, I, C, R, D = ret.T
    R_0_over_time = [beta(i)/gamma for i in range(len(t))]

    return t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D


def deriv(y, t, beta, gamma, sigma, N, p_I_to_C, p_C_to_D, Beds):
    S, E, I, C, R, D = y
    dSdt = -beta(t) * I * S / N
    dEdt = beta(t) * I * S / N - sigma * E
    dIdt = sigma * E - 1/12.0 * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
    dCdt = 1/12.0 * p_I_to_C * I - 1/7.5 * p_C_to_D * \
        min(Beds(t), C) - max(0, C-Beds(t)) - \
        (1 - p_C_to_D) * 1/6.5 * min(Beds(t), C)
    dRdt = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * \
        1/6.5 * min(Beds(t), C)
    dDdt = 1/7.5 * p_C_to_D * min(Beds(t), C) + max(0, C-Beds(t))
    return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt


def get_data_model(df, date):
    df_train = df[df["date"] <= date]
    return df_train


# Loading Data
df_muertes = pd.read_csv("fallecidos_covid.csv", encoding="latin-1")
df_muertes = get_death_data(df_muertes)
df_muertes_train = get_data_model(df_muertes, datetime.datetime(2020, 6, 15))
data_muerte_train = df_muertes_train["count_acum"].values

data_muerte = df_muertes["count_acum"].values
fig_muertes = px.line(df_muertes, x='date',
                      y='count_acum', width=1000, height=350)

agegroups = pd.read_csv("grupo_edades.csv")
list_columns_age = list(agegroups.columns)
list_columns_age.remove("Location")
agegroup_lookup = dict(
    zip(agegroups['Location'], agegroups[list_columns_age].values))
agegroups = agegroup_lookup["Peru"]
df_age = pd.DataFrame()
df_age["age_groups"] = list_columns_age
df_age["count"] = agegroups


# Graph
st.title('Covid-19 Peru Simulation')

st.header("Model Traning stats")
st.write("Cumulative number of death by coronavirus:", data_muerte[-1])
#st.plotly_chart(fig_muertes, use_container_width=True)


fig_age = px.bar(df_age, x='age_groups', y='count')
st.write("Population of:", agegroups.sum())
#st.plotly_chart(fig_age, use_container_width=True)


# Setting vars
total_uci_beds = 1323
gamma_opt = 1.0/9.0
sigma_opt = 1.0/3.0
init_date = datetime.date(2020, 1, 1)
forecast_date = datetime.date(2020, 10, 1)

# Sidebar
d = init_date
d_prediction = st.sidebar.date_input("End day to forecast:", forecast_date)
uci_beds = st.sidebar.number_input(
    "UCI beds", min_value=0, value=total_uci_beds)
gamma = st.sidebar.number_input("gamma value", min_value=0.0, value=gamma_opt)
sigma = st.sidebar.number_input("sigma value", min_value=0.0, value=sigma_opt)


# Model fit until available data
# vars until 15/06
uci_beds_norm_opt = (total_uci_beds/agegroups.sum())*100_000
r_0_start_opt = 3.042465535210805
k_opt = 0.17910498616203066
x0_opt = 91.82132491018402
r_0_end_opt = 1.0171737426718321
prob_i_to_c_opt = 0.03379128457145741
prob_c_to_d_opt = 0.27494757505407275
s_opt = 0.002544791982378995
# vars until 22/06

#uci_beds_norm_opt = (total_uci_beds/agegroups.sum())*100_000
#r_0_start_opt = 3.000441288575001
#k_opt = 0.8266948066971348
#x0_opt = 94.2643904170116
#r_0_end_opt = 1.0558837889971247
#prob_i_to_c_opt = 0.03173422326675246
#prob_c_to_d_opt = 0.28819472293468085
#s_opt = 0.003540006019919522


# Optimized params with ML
r_0_start = st.sidebar.slider("R_0_start:", 3.0, 10.0, r_0_start_opt)
k = st.sidebar.slider("k:", 0.01, 10.0, k_opt)
x0 = st.sidebar.slider("x0:", 0.0, 180.0, x0_opt)
r_0_end = st.sidebar.slider("R_0_end:", 0.1, 4.0, r_0_end_opt)
prob_i_to_c = st.sidebar.slider("prob_I_to_C:", 0.01, 0.1, prob_i_to_c_opt)
prob_c_to_d = st.sidebar.slider("prob_C_to_D:", 0.05, 0.8, prob_c_to_d_opt)
s = st.sidebar.slider("s:", 0.0009, 0.01, s_opt)


d_delta = df_muertes["date"].iloc[-1].date() - d
days = d_delta.days
x_data = np.linspace(0, days - 1, days, dtype=int)
x_ticks = pd.date_range(start=d, periods=len(x_data), freq="D")
uci_beds_norm = (uci_beds/agegroups.sum())*100_000
y_data = np.concatenate((np.zeros(days - len(data_muerte)), data_muerte))
y_data_train = np.concatenate(
    (np.zeros(days - len(data_muerte)), data_muerte_train))

d_delta_prediction = d_prediction - d
days_prediction = d_delta_prediction.days

t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D = Model(days,
                                                                           agegroups,
                                                                           uci_beds_norm_opt,
                                                                           r_0_start_opt,
                                                                           k_opt,
                                                                           x0_opt,
                                                                           r_0_end_opt,
                                                                           prob_i_to_c_opt,
                                                                           prob_c_to_d_opt,
                                                                           s_opt, gamma_opt, sigma_opt)

t_, S_, E_, I_, C_, R_, D_, R_0_over_time_, Beds_, prob_I_to_C_, prob_C_to_D_ = Model(days,
                                                                                      agegroups,
                                                                                      uci_beds_norm,
                                                                                      r_0_start,
                                                                                      k,
                                                                                      x0,
                                                                                      r_0_end,
                                                                                      prob_i_to_c,
                                                                                      prob_c_to_d,
                                                                                      s, gamma, sigma)
mean_error = round(abs(sum([(y_data[i]-D[i]) for i in range(days)])/days), 2)
mean_error_ = round(abs(sum([(y_data[i]-D_[i]) for i in range(days)])/days), 2)
st.write("Model Death Estimation train from ", init_date, " until ",
         df_muertes_train["date"].iloc[-1].date(), "with mean error of:", mean_error)
st.write("Your model have this mean error:", mean_error_)

df_real = pd.DataFrame()
df_est = pd.DataFrame()
df_train = pd.DataFrame()
df_est_ = pd.DataFrame()

df_real["x"] = x_ticks
df_real["value"] = y_data
df_real["type"] = "real"

df_est["x"] = x_ticks
df_est["value"] = D
df_est["type"] = "best params estimation"

df_train["x"] = x_ticks[:len(y_data_train)]
df_train["value"] = y_data_train
df_train["type"] = "train values"

df_est_["x"] = x_ticks
df_est_["value"] = D_
df_est_["type"] = "New model estimation"

df_compare = pd.concat([df_real, df_est, df_train, df_est_], axis=0)

fig_model = px.line(df_compare, x='x', y='value',
                    width=1000, height=350, color="type")
st.plotly_chart(fig_model, use_container_width=True)

df_latent = pd.DataFrame()
df_latent["x"] = x_ticks
df_latent["value"] = E_
df_latent["type"] = "Exposed"

fig_model_ = px.line(df_latent, x='x', y='value',
                     width=1000, height=350, color="type")
st.plotly_chart(fig_model_, use_container_width=True)


# Model Predictions until users specification
st.header("Model Forecasting stats")
x_ticks = pd.date_range(start=d, periods=days_prediction, freq="D")
t, S, E, I, C, R, D, R_0_over_time, Beds, prob_I_to_C, prob_C_to_D = Model(days_prediction,
                                                                           agegroups,
                                                                           uci_beds_norm,
                                                                           r_0_start,
                                                                           k,
                                                                           x0,
                                                                           r_0_end,
                                                                           prob_i_to_c,
                                                                           prob_c_to_d,
                                                                           s, gamma, sigma)

fig_i_r = make_subplots(rows=2,
                        cols=2,
                        subplot_titles=(
                            "Number of Death by Covid-19", "Number of infected people", "Number of recovered people"),
                        specs=[[{"colspan": 2}, None], [{}, {}]],
                        )
fig_i_r.add_trace(
    go.Scatter(x=x_ticks[:len(y_data)], y=y_data),
    row=1, col=1
)
fig_i_r.add_trace(
    go.Scatter(x=x_ticks, y=D),
    row=1, col=1
)
fig_i_r.add_trace(
    go.Scatter(x=x_ticks, y=I),
    row=2, col=1
)
fig_i_r.add_trace(
    go.Scatter(x=x_ticks, y=R),
    row=2, col=2
)
fig_i_r.update_layout(showlegend=False, width=1000, height=600)
st.plotly_chart(fig_i_r, use_container_width=True)


total_CFR = [0] + [100 * D[i] /
                   sum(sigma*E[:i]) if sum(sigma*E[:i]) > 0 else 0 for i in range(1, len(t))]
daily_CFR = [0] + [100 * ((D[i]-D[i-1]) / ((R[i]-R[i-1]) + (D[i]-D[i-1])))
                   if max((R[i]-R[i-1]), (D[i]-D[i-1])) > 10 else 0 for i in range(1, len(t))]
newDs = [0] + [D[i]-D[i-1] for i in range(1, len(t))]
over_capacity_death = [max(0, C[i]-Beds(i)) for i in range(len(t))]
fig_r_fr_uci = make_subplots(rows=1, cols=3, subplot_titles=("R value over time",
                                                             "Fatality Rate %",
                                                             "Death overcapacity UCI"
                                                             ))
fig_r_fr_uci.add_trace(
    go.Scatter(x=x_ticks, y=R_0_over_time),
    row=1, col=1
)
fig_r_fr_uci.add_trace(
    go.Scatter(x=x_ticks, y=daily_CFR),
    row=1, col=2
)
fig_r_fr_uci.add_trace(
    go.Scatter(x=x_ticks, y=total_CFR),
    row=1, col=2
)
fig_r_fr_uci.add_trace(
    go.Scatter(x=x_ticks, y=over_capacity_death),
    row=1, col=3
)
fig_r_fr_uci.add_trace(
    go.Scatter(x=x_ticks, y=newDs),
    row=1, col=3
)

fig_r_fr_uci.update_layout(showlegend=False, width=1000, height=400)
st.plotly_chart(fig_r_fr_uci, use_container_width=True)
