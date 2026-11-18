"""
DST Airlines - Livrable 5
Dashboard interactif : visualisation des vols et prédiction de retard.

Pré-requis : avoir lancé train_model.py au préalable (le prédicteur utilise
le modèle entraîné models/xgb_classifier.pkl ; sans lui, un score heuristique
de secours est utilisé et signalé comme tel).

Lancement sur la VM : python3 dashboard.py
Accès depuis le poste local via tunnel SSH :
  ssh -i jan25_alt2_de_airlines.pem -L 8050:localhost:8050 ubuntu@35.181.54.145
puis ouvrir http://localhost:8050
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

warnings.filterwarnings('ignore')

POSTGRES_URI = 'postgresql://dst_admin:dst_password123@localhost:5432/dst_db'
MODEL_PATH   = '../ml_engine/models/xgb_classifier.pkl'
META_PATH    = '../ml_engine/models/model_meta.json'
METRICS_PATH = '../ml_engine/model_metrics.json'

# Thème pastel clair. Les situations normales utilisent des tons doux ;
# l'orange et le rouge sont réservés aux alertes (retards, risque élevé).
PALETTE = {
    'bg':      '#f6f7fb',   # fond général
    'surface': '#ffffff',   # bandeau
    'card':    '#ffffff',   # cartes
    'border':  '#e3e7f0',
    'accent':  '#7c9cd9',   # bleu pastel
    'green':   '#8fd3b6',   # vert pastel (situation normale)
    'red':     '#e35d5b',   # rouge (alerte)
    'yellow':  '#ef9b4f',   # orange (vigilance)
    'text':    '#33415c',
    'muted':   '#8a93a6',
}

# Échelle commune : vert pastel -> orange -> rouge (les problèmes ressortent)
SCALE_ALERT = ['#8fd3b6', '#ef9b4f', '#e35d5b']

DAY_NAMES   = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
MONTH_NAMES = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
               'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
CONDITIONS  = ['Clear', 'Cloudy', 'Rainy', 'Foggy', 'Snowy', 'Stormy']


# ------------------------------------------------------------------
# Chargement des données et du modèle
# ------------------------------------------------------------------

def find_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data():
    try:
        engine = create_engine(POSTGRES_URI)
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM gold_ml_features;", conn)
        print(f'{len(df):,} vols chargés depuis PostgreSQL')
        return df
    except Exception as e:
        print(f'BDD indisponible ({e}) : données de démonstration')
        return generate_demo_data()


def generate_demo_data(n=5000):
    """Jeu de secours pour développer le dashboard hors de la VM."""
    rng = np.random.default_rng(42)
    origins = ['LHR', 'AMS', 'MAD', 'FCO', 'FRA', 'BCN', 'LIS', 'MXP', 'VIE', 'ZRH']
    ts = pd.date_range('2025-01-01', '2025-09-30', periods=n)
    df = pd.DataFrame({
        'arrival_ts':          ts,
        'arrival_hour':        ts.hour,
        'day_of_week':         ts.dayofweek,
        'month':               ts.month,
        'is_weekend':          (ts.dayofweek >= 5).astype(int),
        'is_holiday':          rng.choice([0, 1], n, p=[0.95, 0.05]),
        'visibility_m':        rng.uniform(200, 10000, n),
        'wind_gusts_kmh':      rng.gamma(3, 9, n),
        'wind_kmh':            rng.gamma(3, 5, n),
        'temp_c':              rng.normal(13, 8, n),
        'condition':           rng.choice(CONDITIONS, n, p=[.35, .30, .20, .07, .04, .04]),
        'hourly_flight_count': rng.poisson(28, n),
        'daily_flight_count':  rng.poisson(600, n),
        'duration_min':        rng.uniform(60, 240, n),
        'origin_airport':      rng.choice(origins, n),
        'icao24':              rng.choice([f'39{i:04x}' for i in range(300)], n),
        'delay_minutes':       rng.normal(4, 6, n),
    })
    return df


def build_display_targets(df, C):
    """Reconstruit la cible affichée par le dashboard avec la même logique
    de risque que train_model.py (calibrage ~23%, référence Eurocontrol)."""
    np.random.seed(42)
    risk = np.zeros(len(df))
    if C['visibility']:
        risk += (df[C['visibility']] < 600).astype(float) * 0.40
        risk += ((df[C['visibility']] >= 600) & (df[C['visibility']] < 1500)).astype(float) * 0.20
    if C['gusts']:
        risk += (df[C['gusts']] > 70).astype(float) * 0.25
        risk += ((df[C['gusts']] > 50) & (df[C['gusts']] <= 70)).astype(float) * 0.10
    if C['condition']:
        risk += df[C['condition']].isin(['Foggy', 'Snowy', 'Stormy']).astype(float) * 0.15
    if C['hourly']:
        q75, q90 = df[C['hourly']].quantile(0.75), df[C['hourly']].quantile(0.90)
        risk += (df[C['hourly']] >= q90).astype(float) * 0.30
        risk += ((df[C['hourly']] >= q75) & (df[C['hourly']] < q90)).astype(float) * 0.15
    if C['holiday']:
        risk += df[C['holiday']].astype(float) * 0.10
    if C['weekend']:
        risk += df[C['weekend']].astype(float) * 0.05

    df['is_delayed'] = (risk >= np.percentile(risk, 77)).astype(int)

    base = df[C['delay']].values if C['delay'] else np.zeros(len(df))
    noise = np.where(df['is_delayed'] == 1,
                     np.random.exponential(scale=18, size=len(df)) + 10,
                     np.random.uniform(-8, 12, size=len(df)))
    df['delay_min'] = np.clip(base + noise * df['is_delayed'] * 0.7, -15, 120)
    return df


def load_model():
    model, meta = None, None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            with open(META_PATH) as f:
                meta = json.load(f)
            print('Modèle XGBoost chargé')
        except Exception as e:
            print(f'Modèle illisible ({e}) : prédicteur en mode heuristique')
    else:
        print('../ml_engine/models/xgb_classifier.pkl absent : prédicteur en mode heuristique')
    return model, meta


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


# Chargement initial
df = load_data()

C = {
    'ts':         find_col(df, 'arrival_ts', 'timestamp'),
    'hour':       find_col(df, 'arrival_hour', 'fi_arrival_hour'),
    'visibility': find_col(df, 'visibility_m', 'w_visibility_m'),
    'gusts':      find_col(df, 'wind_gusts_kmh', 'w_wind_gusts_kmh'),
    'wind':       find_col(df, 'wind_kmh', 'w_wind_kmh'),
    'temp':       find_col(df, 'temp_c', 'w_temp_c'),
    'condition':  find_col(df, 'condition', 'w_condition'),
    'hourly':     find_col(df, 'hourly_flight_count', 'ctx_hourly_flight_count'),
    'daily':      find_col(df, 'daily_flight_count', 'ctx_daily_flight_count'),
    'dow':        find_col(df, 'day_of_week', 't_day_of_week'),
    'month':      find_col(df, 'month', 't_month'),
    'weekend':    find_col(df, 'is_weekend', 't_is_weekend'),
    'holiday':    find_col(df, 'is_holiday', 't_is_holiday'),
    'duration':   find_col(df, 'duration_min', 'fi_duration_min'),
    'origin':     find_col(df, 'origin_airport', 'metadata.origin_airport'),
    'icao24':     find_col(df, 'icao24', 'metadata.icao24'),
    'delay':      find_col(df, 'delay_minutes', 'fi_delay_minutes'),
}
if C['ts']:
    df[C['ts']] = pd.to_datetime(df[C['ts']])

df = build_display_targets(df, C)
model, meta = load_model()
metrics = load_metrics()


# ------------------------------------------------------------------
# Application Dash
# ------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title='DST Airlines - ML Dashboard',
)


def kpi_card(icon, value, label, color=PALETTE['accent']):
    return dbc.Card([
        dbc.CardBody([
            html.I(className=f'fa {icon} fa-2x', style={'color': color}),
            html.H3(value, style={'color': PALETTE['text'], 'margin': '8px 0 2px'}),
            html.P(label, style={'color': PALETTE['muted'], 'fontSize': '13px',
                                 'marginBottom': 0}),
        ])
    ], style={'background': PALETTE['card'],
              'border': f'1px solid {PALETTE["border"]}',
              'borderTop': f'3px solid {color}',
              'borderRadius': '8px', 'textAlign': 'center',
              'boxShadow': '0 1px 4px rgba(51, 65, 92, 0.06)'})


def wrap_graph(fig, height=350):
    return dbc.Card(
        [dcc.Graph(figure=fig, config={'displayModeBar': False},
                   style={'height': f'{height}px'})],
        style={'background': PALETTE['card'],
               'border': f'1px solid {PALETTE["border"]}',
               'borderRadius': '8px'})


def dark_layout(fig, **kwargs):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color=PALETTE['text'],
        margin=dict(t=45, b=30, l=40, r=15), **kwargs)
    fig.update_xaxes(gridcolor='#e8ecf4', zerolinecolor='#e3e7f0')
    fig.update_yaxes(gridcolor='#e8ecf4', zerolinecolor='#e3e7f0')
    return fig


app.layout = html.Div(
    style={'backgroundColor': PALETTE['bg'], 'minHeight': '100vh',
           'fontFamily': '"Inter", sans-serif'},
    children=[
        html.Div(
            style={'background': PALETTE['surface'],
                   'borderBottom': f'1px solid {PALETTE["border"]}',
                   'padding': '18px 30px', 'display': 'flex',
                   'alignItems': 'center', 'justifyContent': 'space-between'},
            children=[
                html.Div([
                    html.Div([
                        html.I(className='fa fa-plane',
                               style={'color': PALETTE['accent'],
                                      'fontSize': '22px', 'marginRight': '10px'}),
                        html.Span('DST Airlines',
                                  style={'fontSize': '20px', 'fontWeight': 'bold',
                                         'color': PALETTE['text']}),
                        html.Span(' - ML Dashboard',
                                  style={'fontSize': '15px',
                                         'color': PALETTE['muted']}),
                    ]),
                    html.P('Livrable 5 - Prédiction de retard des vols (arrivées CDG)',
                           style={'color': PALETTE['muted'], 'fontSize': '12px',
                                  'margin': '4px 0 0'}),
                ]),
                html.Span(f'{len(df):,} vols analysés',
                          style={'background': PALETTE['accent'] + '22',
                                 'color': PALETTE['accent'],
                                 'padding': '6px 14px', 'borderRadius': '20px',
                                 'fontSize': '13px',
                                 'border': f'1px solid {PALETTE["accent"]}44'}),
            ]),

        dbc.Tabs(id='main-tabs', active_tab='tab-overview', children=[
            dbc.Tab(label="Vue d'ensemble", tab_id='tab-overview',
                    label_style={'color': PALETTE['muted']},
                    active_label_style={'color': PALETTE['accent']}),
            dbc.Tab(label='Analyse des retards', tab_id='tab-delays',
                    label_style={'color': PALETTE['muted']},
                    active_label_style={'color': PALETTE['accent']}),
            dbc.Tab(label='Performances ML', tab_id='tab-ml',
                    label_style={'color': PALETTE['muted']},
                    active_label_style={'color': PALETTE['accent']}),
            dbc.Tab(label='Prédicteur', tab_id='tab-predict',
                    label_style={'color': PALETTE['muted']},
                    active_label_style={'color': PALETTE['accent']}),
            dbc.Tab(label='Temps réel', tab_id='tab-live',
                    label_style={'color': PALETTE['muted']},
                    active_label_style={'color': PALETTE['accent']}),
        ]),

        html.Div(id='tab-content', style={'padding': '24px 30px'}),
    ])


@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'active_tab'))
def render_tab(tab):
    if tab == 'tab-overview':
        return render_overview()
    if tab == 'tab-delays':
        return render_delays()
    if tab == 'tab-ml':
        return render_ml()
    if tab == 'tab-predict':
        return render_predictor()
    if tab == 'tab-live':
        return render_live()
    return html.Div('Onglet non trouvé')


# ------------------------------------------------------------------
# Onglet 1 : Vue d'ensemble
# ------------------------------------------------------------------

def render_overview():
    total     = len(df)
    n_delayed = int(df['is_delayed'].sum())
    rate      = n_delayed / total * 100
    avg_delay = df.loc[df['delay_min'] > 0, 'delay_min'].mean()

    kpi_row = dbc.Row([
        dbc.Col(kpi_card('fa-plane-arrival', f'{total:,}', 'Vols analysés'), md=3),
        dbc.Col(kpi_card('fa-clock', f'{n_delayed:,}', 'Vols en retard',
                         PALETTE['red']), md=3),
        dbc.Col(kpi_card('fa-percent', f'{rate:.1f}%', 'Taux de retard',
                         PALETTE['yellow']), md=3),
        dbc.Col(kpi_card('fa-hourglass-half', f'{avg_delay:.0f} min',
                         'Retard moyen (vols retardés)', PALETTE['green']), md=3),
    ], className='mb-4')

    fig_pie = go.Figure(go.Pie(
        labels=["À l'heure", 'En retard'],
        values=[total - n_delayed, n_delayed], hole=0.65,
        marker_colors=[PALETTE['green'], PALETTE['red']],
        textinfo='percent+label', textfont_color=PALETTE['text']))
    dark_layout(fig_pie, title='Répartition des vols', showlegend=False)
    fig_pie.update_layout(annotations=[dict(
        text=f'<b>{rate:.0f}%</b><br>retard', font_size=16,
        font_color=PALETTE['text'], showarrow=False)])

    if C['hour']:
        hourly = df.groupby(C['hour'])['is_delayed'].mean().reset_index()
        fig_hour = px.bar(hourly, x=C['hour'], y='is_delayed',
                          color='is_delayed',
                          color_continuous_scale=SCALE_ALERT,
                          title="Taux de retard par heure d'arrivée")
        dark_layout(fig_hour, xaxis_title='Heure', yaxis_title='Taux de retard',
                    showlegend=False)
        fig_hour.update_coloraxes(showscale=False)
    else:
        fig_hour = dark_layout(go.Figure())

    fig_hist = px.histogram(
        df[df['delay_min'].between(-30, 120)], x='delay_min', nbins=80,
        color_discrete_sequence=[PALETTE['accent']],
        title="Distribution des retards à l'arrivée (minutes)")
    fig_hist.add_vline(x=15, line_dash='dash', line_color=PALETTE['red'],
                       annotation_text='Seuil IATA 15 min',
                       annotation_font_color=PALETTE['red'])
    dark_layout(fig_hist, xaxis_title='Retard (min)', yaxis_title='Nb vols')

    return html.Div([
        kpi_row,
        dbc.Row([dbc.Col(wrap_graph(fig_pie), md=4),
                 dbc.Col(wrap_graph(fig_hour), md=8)], className='mb-4'),
        dbc.Row([dbc.Col(wrap_graph(fig_hist), md=12)]),
    ])


# ------------------------------------------------------------------
# Onglet 2 : Analyse des retards
# ------------------------------------------------------------------

def render_delays():
    rows = []

    if C['hour'] and C['dow']:
        heat = (df.groupby([C['dow'], C['hour']])['is_delayed']
                  .mean().reset_index()
                  .pivot(index=C['dow'], columns=C['hour'], values='is_delayed'))
        heat.index = [DAY_NAMES[int(i)] for i in heat.index]
        fig_heat = go.Figure(go.Heatmap(
            z=heat.values,
            x=[f'{h}h' for h in heat.columns], y=heat.index,
            colorscale=[[0, SCALE_ALERT[0]], [0.5, SCALE_ALERT[1]], [1, SCALE_ALERT[2]]],
            hovertemplate='%{y} %{x}<br>Retard : %{z:.1%}<extra></extra>'))
        dark_layout(fig_heat, title='Taux de retard par jour et heure')
        rows.append(dbc.Row([dbc.Col(wrap_graph(fig_heat, 380), md=12)],
                            className='mb-4'))

    if C['condition']:
        cond = (df.groupby(C['condition'])
                  .agg(taux=('is_delayed', 'mean'), nb=('is_delayed', 'count'))
                  .reindex(CONDITIONS).dropna().reset_index())
        cond['taux_pct'] = cond['taux'] * 100
        fig_cond = px.bar(cond, x=C['condition'], y='taux_pct', text='taux_pct',
                          color='taux_pct',
                          color_continuous_scale=SCALE_ALERT,
                          title='Taux de retard par condition météo')
        fig_cond.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        dark_layout(fig_cond, xaxis_title='', yaxis_title='% retards')
        fig_cond.update_coloraxes(showscale=False)
        rows.append(dbc.Row([dbc.Col(wrap_graph(fig_cond), md=6),
                             dbc.Col(wrap_graph(visibility_fig()), md=6)],
                            className='mb-4'))

    if C['month']:
        monthly = df.groupby(C['month'])['is_delayed'].mean().reset_index()
        monthly['mois'] = monthly[C['month']].apply(lambda m: MONTH_NAMES[int(m) - 1])
        fig_month = px.line(monthly, x='mois', y='is_delayed', markers=True,
                            color_discrete_sequence=[PALETTE['accent']],
                            title='Évolution mensuelle du taux de retard')
        dark_layout(fig_month, xaxis_title='', yaxis_title='Taux de retard')
        rows.append(dbc.Row([dbc.Col(wrap_graph(fig_month), md=6),
                             dbc.Col(wrap_graph(origin_fig()), md=6)],
                            className='mb-4'))

    return html.Div(rows) if rows else html.P(
        'Colonnes insuffisantes pour cette analyse',
        style={'color': PALETTE['muted']})


def visibility_fig():
    if not C['visibility']:
        return dark_layout(go.Figure())
    bins = [0, 600, 1500, 3000, 5000, 8000, 100000]
    labels = ['< 600 m (LVP)', '600-1500', '1500-3000', '3000-5000',
              '5000-8000', '> 8000 m']
    tmp = df.copy()
    tmp['vis_cat'] = pd.cut(tmp[C['visibility']], bins=bins, labels=labels)
    vis = tmp.groupby('vis_cat')['is_delayed'].mean().reset_index()
    fig = px.bar(vis, x='vis_cat', y='is_delayed',
                 color_discrete_sequence=[PALETTE['accent']],
                 title='Taux de retard selon la visibilité')
    dark_layout(fig, xaxis_title='', yaxis_title='Taux de retard')
    return fig


def origin_fig():
    if not C['origin']:
        return dark_layout(go.Figure())
    stats = (df.groupby(C['origin'])
               .agg(taux=('is_delayed', 'mean'), nb=('is_delayed', 'count'))
               .reset_index())
    stats = stats[stats['nb'] >= 30].sort_values('taux', ascending=False).head(10)
    stats['taux_pct'] = stats['taux'] * 100
    fig = px.bar(stats, x='taux_pct', y=C['origin'], orientation='h',
                 text='taux_pct', color='taux_pct',
                 color_continuous_scale=SCALE_ALERT,
                 title='Origines les plus touchées par les retards (min. 30 vols)')
    fig.update_traces(texttemplate='%{text:.1f}%')
    dark_layout(fig, xaxis_title='% retards', yaxis_title='')
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(autorange='reversed')
    return fig


# ------------------------------------------------------------------
# Onglet 3 : Performances ML
# ------------------------------------------------------------------

def render_ml():
    if not metrics:
        return dbc.Alert(
            'model_metrics.json introuvable. Lancer train_model.py pour '
            'entraîner les modèles et générer les métriques.',
            color='warning')

    clf = metrics['classification']
    model_names = list(clf.keys())
    keys   = ['f1', 'precision', 'recall', 'accuracy', 'auc_roc']
    labels = ['F1-score', 'Précision', 'Rappel', 'Accuracy', 'AUC-ROC']
    colors = [PALETTE['accent'], PALETTE['green'], PALETTE['yellow'],
              '#8b5cf6', '#ec4899']

    best_name = max(model_names, key=lambda m: clf[m].get('f1', 0))
    best = clf[best_name]

    best_card = dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className='fa fa-trophy fa-2x',
                       style={'color': PALETTE['yellow'], 'marginRight': '14px'}),
                html.H4(f'Meilleur modèle : {best_name}',
                        style={'color': PALETTE['text'], 'margin': 0}),
            ], style={'display': 'flex', 'alignItems': 'center',
                      'marginBottom': '16px'}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H3(f'{best["f1"]:.4f}', style={'color': PALETTE['accent']}),
                    html.P('F1-score', style={'color': PALETTE['muted']})]), md=3),
                dbc.Col(html.Div([
                    html.H3(f'{best["auc_roc"]:.4f}', style={'color': PALETTE['green']}),
                    html.P('AUC-ROC', style={'color': PALETTE['muted']})]), md=3),
                dbc.Col(html.Div([
                    html.H3(f'{best["accuracy"]:.4f}', style={'color': PALETTE['yellow']}),
                    html.P('Accuracy', style={'color': PALETTE['muted']})]), md=3),
                dbc.Col(html.Div([
                    html.H3('RandomizedSearchCV',
                            style={'color': PALETTE['text'], 'fontSize': '20px'}),
                    html.P('30 itérations, TimeSeriesSplit 5 folds',
                           style={'color': PALETTE['muted']})]), md=3),
            ]),
        ])
    ], style={'background': PALETTE['card'],
              'border': f'2px solid {PALETTE["accent"]}',
              'borderRadius': '10px', 'marginBottom': '24px'})

    fig_compare = go.Figure()
    for met, lab, col in zip(keys, labels, colors):
        fig_compare.add_trace(go.Bar(
            name=lab, x=model_names,
            y=[clf[m].get(met, 0) for m in model_names],
            marker_color=col,
            text=[f'{clf[m].get(met, 0):.3f}' for m in model_names],
            textposition='outside'))
    dark_layout(fig_compare, barmode='group',
                title='Comparaison des modèles (jeu de test, Septembre)',
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                yaxis=dict(range=[0, 1.05]))

    fig_radar = go.Figure()
    for m in model_names:
        vals = [clf[m].get(k, 0) for k in keys]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=labels + [labels[0]],
            fill='toself', name=m, opacity=0.5))
    dark_layout(fig_radar, title='Profil des modèles',
                polar=dict(radialaxis=dict(visible=True, range=[0, 1]),
                           bgcolor='rgba(0,0,0,0)'),
                legend=dict(bgcolor='rgba(0,0,0,0)'))

    table_data = []
    for m in model_names:
        row = {'Modèle': m}
        for k, lab in zip(keys, labels):
            row[lab] = f'{clf[m].get(k, 0):.4f}'
        table_data.append(row)

    table = dash_table.DataTable(
        data=table_data,
        columns=[{'name': c, 'id': c} for c in table_data[0]],
        style_header={'backgroundColor': PALETTE['card'],
                      'color': PALETTE['text'], 'fontWeight': 'bold',
                      'border': f'1px solid {PALETTE["border"]}'},
        style_data={'backgroundColor': PALETTE['surface'],
                    'color': PALETTE['text'],
                    'border': f'1px solid {PALETTE["border"]}'},
        style_data_conditional=[{
            'if': {'filter_query': f'{{Modèle}} = "{best_name}"'},
            'backgroundColor': PALETTE['accent'] + '22',
            'fontWeight': 'bold'}],
        style_table={'overflowX': 'auto'})

    reg = metrics.get('regression', {})
    reg_block = html.Div()
    if reg:
        best_reg = metrics.get('best_regressor', list(reg.keys())[0])
        reg_block = dbc.Card([dbc.CardBody([
            html.H6('Régression du retard (minutes)',
                    style={'color': PALETTE['text']}),
            html.P(f'{best_reg} : MAE = {reg[best_reg]["mae"]:.2f} min, '
                   f'RMSE = {reg[best_reg]["rmse"]:.2f} min, '
                   f'R² = {reg[best_reg]["r2"]:.4f}',
                   style={'color': PALETTE['muted'], 'marginBottom': 0}),
        ])], style={'background': PALETTE['card'],
                    'border': f'1px solid {PALETTE["border"]}',
                    'borderRadius': '8px', 'marginTop': '20px'})

    return html.Div([
        best_card,
        dbc.Row([dbc.Col(wrap_graph(fig_compare, 400), md=8),
                 dbc.Col(wrap_graph(fig_radar, 400), md=4)], className='mb-4'),
        dbc.Card([dbc.CardBody([
            html.H6('Tableau comparatif complet',
                    style={'color': PALETTE['text'], 'marginBottom': '12px'}),
            table])],
            style={'background': PALETTE['card'],
                   'border': f'1px solid {PALETTE["border"]}',
                   'borderRadius': '8px'}),
        reg_block,
    ])


# ------------------------------------------------------------------
# Onglet 4 : Prédicteur
# ------------------------------------------------------------------

def render_predictor():
    origins = (sorted(df[C['origin']].dropna().unique().tolist())
               if C['origin'] else [])
    label_style = {'color': PALETTE['muted'], 'fontSize': '13px',
                   'marginBottom': '4px'}

    status = ('Prédiction par le modèle XGBoost entraîné' if model is not None
              else 'Modèle non chargé : score heuristique de secours '
                   '(lancer train_model.py)')
    status_color = PALETTE['green'] if model is not None else PALETTE['yellow']

    form = dbc.Card([
        dbc.CardHeader(html.H5('Simuler une prédiction de retard',
                               style={'color': PALETTE['text'], 'margin': 0}),
                       style={'background': PALETTE['card'],
                              'borderBottom': f'1px solid {PALETTE["border"]}'}),
        dbc.CardBody([
            html.Small(status, style={'color': status_color,
                                      'display': 'block',
                                      'marginBottom': '16px'}),
            dbc.Row([
                dbc.Col([html.Label("Heure d'arrivée", style=label_style),
                         dcc.Slider(id='pred-hour', min=0, max=23, step=1, value=8,
                                    marks={i: f'{i}h' for i in [0, 6, 12, 18, 23]})],
                        md=4),
                dbc.Col([html.Label('Jour de la semaine', style=label_style),
                         dcc.Dropdown(id='pred-day',
                                      options=[{'label': d, 'value': i}
                                               for i, d in enumerate(DAY_NAMES)],
                                      value=0, className='dark-dropdown')], md=4),
                dbc.Col([html.Label('Mois', style=label_style),
                         dcc.Dropdown(id='pred-month',
                                      options=[{'label': m, 'value': i + 1}
                                               for i, m in enumerate(MONTH_NAMES)],
                                      value=6, className='dark-dropdown')], md=4),
            ], className='mb-3'),
            dbc.Row([
                dbc.Col([html.Label('Condition météo', style=label_style),
                         dcc.Dropdown(id='pred-condition',
                                      options=[{'label': c, 'value': c}
                                               for c in CONDITIONS],
                                      value='Clear', className='dark-dropdown')],
                        md=4),
                dbc.Col([html.Label('Visibilité (m)', style=label_style),
                         dcc.Slider(id='pred-visibility', min=100, max=10000,
                                    step=100, value=8000,
                                    marks={100: '100', 600: '600', 5000: '5000',
                                           10000: '10000'})], md=4),
                dbc.Col([html.Label('Rafales de vent (km/h)', style=label_style),
                         dcc.Slider(id='pred-gusts', min=0, max=110, step=5,
                                    value=20,
                                    marks={0: '0', 50: '50', 70: '70',
                                           110: '110'})], md=4),
            ], className='mb-3'),
            dbc.Row([
                dbc.Col([html.Label('Trafic horaire à CDG (nb vols)',
                                    style=label_style),
                         dcc.Slider(id='pred-traffic', min=5, max=70, step=1,
                                    value=25,
                                    marks={5: '5', 25: '25', 45: '45',
                                           70: '70'})], md=6),
                dbc.Col([html.Label("Aéroport d'origine", style=label_style),
                         dcc.Dropdown(id='pred-origin',
                                      options=[{'label': o, 'value': o}
                                               for o in origins],
                                      value=origins[0] if origins else None,
                                      className='dark-dropdown')], md=6),
            ], className='mb-3'),
            dbc.Button('Prédire', id='pred-btn', n_clicks=0,
                       style={'backgroundColor': PALETTE['accent'],
                              'border': 'none', 'padding': '10px 30px'}),
        ])
    ], style={'background': PALETTE['card'],
              'border': f'1px solid {PALETTE["border"]}',
              'borderRadius': '8px', 'marginBottom': '24px'})

    return html.Div([form, html.Div(id='pred-result')])


def build_feature_row(hour, day, month, condition, visibility, gusts,
                      traffic, origin):
    """Construit la ligne de features attendue par le pipeline entraîné :
    valeurs par défaut (médianes/modes du train) puis surcharge avec les
    entrées du formulaire et recalcul des features dérivées."""
    row = dict(meta['defaults'])
    cols = meta['columns']

    if cols['hour']:       row[cols['hour']] = hour
    if cols['visibility']: row[cols['visibility']] = visibility
    if cols['gusts']:      row[cols['gusts']] = gusts
    if cols['hourly']:     row[cols['hourly']] = traffic
    if cols['condition']:  row[cols['condition']] = condition
    if cols['origin'] and origin: row[cols['origin']] = origin
    if cols['weekend']:    row[cols['weekend']] = int(day >= 5)

    row['arrival_hour_sin'] = np.sin(2 * np.pi * hour / 24)
    row['arrival_hour_cos'] = np.cos(2 * np.pi * hour / 24)
    row['day_sin']   = np.sin(2 * np.pi * day / 7)
    row['day_cos']   = np.cos(2 * np.pi * day / 7)
    row['month_sin'] = np.sin(2 * np.pi * month / 12)
    row['month_cos'] = np.cos(2 * np.pi * month / 12)
    row['is_lvp']       = int(visibility < 600)
    row['is_high_wind'] = int(gusts > 55)
    row['is_peak_hour'] = int(hour in [6, 7, 8, 9, 17, 18, 19, 20])

    X = pd.DataFrame([row])
    return X[meta['all_features']]


def heuristic_risk(hour, day, month, condition, visibility, gusts, traffic):
    """Score de secours si le modèle n'est pas disponible (même logique
    que la cible d'entraînement, sans calibrage exact)."""
    risk = 0.05
    if visibility < 600:
        risk += 0.40
    elif visibility < 1500:
        risk += 0.20
    if gusts > 70:
        risk += 0.25
    elif gusts > 50:
        risk += 0.10
    if condition in ('Foggy', 'Snowy', 'Stormy'):
        risk += 0.15
    if traffic >= 45:
        risk += 0.30
    elif traffic >= 35:
        risk += 0.15
    if day >= 5:
        risk += 0.05
    if hour in (6, 7, 8, 9, 17, 18, 19, 20):
        risk += 0.05
    return float(min(0.97, max(0.02, risk)))


@app.callback(
    Output('pred-result', 'children'),
    Input('pred-btn', 'n_clicks'),
    State('pred-hour', 'value'),
    State('pred-day', 'value'),
    State('pred-month', 'value'),
    State('pred-condition', 'value'),
    State('pred-visibility', 'value'),
    State('pred-gusts', 'value'),
    State('pred-traffic', 'value'),
    State('pred-origin', 'value'),
    prevent_initial_call=True,
)
def predict_delay(n_clicks, hour, day, month, condition, visibility,
                  gusts, traffic, origin):
    if not n_clicks:
        return html.Div()

    if model is not None and meta is not None:
        X = build_feature_row(hour, day, month, condition, visibility,
                              gusts, traffic, origin)
        proba = float(model.predict_proba(X)[0, 1])
        source = 'Modèle XGBoost optimisé (models/xgb_classifier.pkl)'
    else:
        proba = heuristic_risk(hour, day, month, condition, visibility,
                               gusts, traffic)
        source = 'Score heuristique de secours - entraîner le modèle pour une vraie prédiction'

    delayed = proba > 0.5
    label = 'RETARD PROBABLE' if delayed else "À L'HEURE"
    color = PALETTE['red'] if delayed else PALETTE['green']
    icon  = 'fa-times-circle' if delayed else 'fa-check-circle'

    fig_gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(proba * 100, 1),
        number={'suffix': '%', 'font': {'color': color, 'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': PALETTE['muted']},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30],   'color': 'rgba(143, 211, 182, 0.35)'},
                {'range': [30, 60],  'color': 'rgba(239, 155, 79, 0.35)'},
                {'range': [60, 100], 'color': 'rgba(227, 93, 91, 0.35)'},
            ],
            'threshold': {'line': {'color': color, 'width': 3},
                          'value': proba * 100},
        },
        title={'text': 'Probabilité de retard',
               'font': {'color': PALETTE['text'], 'size': 14}}))
    dark_layout(fig_gauge, height=260)

    recap = (f"Heure : {hour}h | Jour : {DAY_NAMES[day]} | "
             f"Mois : {MONTH_NAMES[month - 1]} | Météo : {condition} | "
             f"Visibilité : {visibility} m | Rafales : {gusts} km/h | "
             f"Trafic : {traffic} vols/h"
             + (f" | Origine : {origin}" if origin else ""))

    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Div([
                    html.I(className=f'fa {icon} fa-3x',
                           style={'color': color, 'marginRight': '18px'}),
                    html.Div([
                        html.H3(label, style={'color': color, 'margin': 0}),
                        html.P(f'Probabilité de retard : {proba * 100:.1f}%',
                               style={'color': PALETTE['muted']}),
                    ]),
                ], style={'display': 'flex', 'alignItems': 'center',
                          'height': '100%'}), md=5),
                dbc.Col(dcc.Graph(figure=fig_gauge,
                                  config={'displayModeBar': False}), md=7),
            ]),
            html.Hr(style={'borderColor': PALETTE['border']}),
            html.Small(recap, style={'color': PALETTE['muted'],
                                     'display': 'block'}),
            html.Small(source, style={'color': PALETTE['muted'],
                                      'display': 'block',
                                      'marginTop': '6px'}),
        ])
    ], style={'background': PALETTE['card'],
              'border': f'2px solid {color}', 'borderRadius': '10px'})


# ------------------------------------------------------------------
# Onglet 5 : Temps réel (vols Kafka -> API -> live_predictions)
# Lecture via le Backend API (port 8001), pas en direct dans PostgreSQL.
# ------------------------------------------------------------------

import requests

BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8001')


def load_live_predictions(limit=50):
    """Lit les prédictions live via le Backend API (centralise les accès données).
    En cas d'indisponibilité du backend, repli direct sur PostgreSQL pour ne pas
    bloquer la démo ; si rien n'est disponible, DataFrame vide."""
    try:
        r = requests.get(f'{BACKEND_URL}/predictions/live',
                         params={'limit': limit}, timeout=5)
        r.raise_for_status()
        data = r.json().get('data', [])
        return pd.DataFrame(data)
    except Exception:
        # Repli : lecture directe PostgreSQL (si le backend n'est pas lancé)
        try:
            engine = create_engine(POSTGRES_URI)
            with engine.connect() as conn:
                return pd.read_sql(
                    "SELECT callsign, received_at, arrival_hour, visibility_m, "
                    "wind_gusts_kmh, is_delayed, delay_proba "
                    "FROM live_predictions ORDER BY received_at DESC "
                    f"LIMIT {int(limit)};", conn)
        except Exception:
            return pd.DataFrame()


def render_live():
    # Le dcc.Interval déclenche le rafraîchissement toutes les 30 secondes
    return html.Div([
        html.Div([
            html.Div([
                html.H5('Vols en temps réel',
                        style={'color': PALETTE['text'], 'margin': 0}),
                html.Small('Flux Kafka -> API de prédiction -> PostgreSQL · '
                           'rafraîchissement automatique toutes les 30 s',
                           style={'color': PALETTE['muted']}),
            ]),
            html.Span(id='live-status', style={
                'background': PALETTE['accent'] + '22', 'color': PALETTE['accent'],
                'padding': '6px 14px', 'borderRadius': '20px', 'fontSize': '13px',
                'border': f'1px solid {PALETTE["accent"]}44'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between',
                  'alignItems': 'center', 'marginBottom': '18px'}),

        dcc.Interval(id='live-interval', interval=30 * 1000, n_intervals=0),

        html.Div(id='live-kpis', className='mb-4'),
        html.Div(id='live-content'),
    ])


@app.callback(
    [Output('live-kpis', 'children'),
     Output('live-content', 'children'),
     Output('live-status', 'children')],
    Input('live-interval', 'n_intervals'),
)
def refresh_live(_):
    df_live = load_live_predictions()

    if df_live.empty:
        msg = dbc.Alert(
            "Aucune prédiction en temps réel pour l'instant. Lancez le flux : "
            "uvicorn api:app (port 8000), puis python consumer.py, "
            "puis python test_producer.py.",
            color='secondary',
            style={'background': PALETTE['card'], 'color': PALETTE['muted'],
                   'border': f'1px solid {PALETTE["border"]}'})
        return html.Div(), msg, 'En attente'

    total = len(df_live)
    n_delayed = int(df_live['is_delayed'].sum())
    rate = n_delayed / total * 100 if total else 0
    last = pd.to_datetime(df_live['received_at'].iloc[0])

    kpis = dbc.Row([
        dbc.Col(kpi_card('fa-bolt', f'{total}', 'Vols traités (récents)'), md=4),
        dbc.Col(kpi_card('fa-triangle-exclamation', f'{n_delayed}',
                         'Retards prédits', PALETTE['red']), md=4),
        dbc.Col(kpi_card('fa-percent', f'{rate:.0f}%', 'Taux de retard prédit',
                         PALETTE['yellow']), md=4),
    ])

    # Tableau des derniers vols, avec mise en évidence des retards
    df_show = df_live.copy()
    df_show['Statut'] = df_show['is_delayed'].map(
        {True: 'RETARD', False: "À l'heure"})
    df_show['Proba'] = (df_show['delay_proba'] * 100).round(1).astype(str) + '%'
    df_show['Reçu à'] = pd.to_datetime(df_show['received_at']).dt.strftime('%H:%M:%S')
    df_show = df_show.rename(columns={
        'callsign': 'Vol', 'arrival_hour': 'Heure arr.',
        'visibility_m': 'Visibilité (m)', 'wind_gusts_kmh': 'Rafales (km/h)'})
    cols = ['Vol', 'Reçu à', 'Heure arr.', 'Visibilité (m)',
            'Rafales (km/h)', 'Statut', 'Proba']

    table = dash_table.DataTable(
        data=df_show[cols].to_dict('records'),
        columns=[{'name': c, 'id': c} for c in cols],
        style_header={'backgroundColor': PALETTE['card'],
                      'color': PALETTE['text'], 'fontWeight': 'bold',
                      'border': f'1px solid {PALETTE["border"]}'},
        style_data={'backgroundColor': PALETTE['surface'],
                    'color': PALETTE['text'],
                    'border': f'1px solid {PALETTE["border"]}'},
        style_data_conditional=[{
            'if': {'filter_query': '{Statut} = "RETARD"'},
            'backgroundColor': PALETTE['red'] + '22',
            'color': PALETTE['text']}],
        style_table={'overflowX': 'auto'},
        page_size=15,
    )
    card = dbc.Card([dbc.CardBody([
        html.H6('Derniers vols reçus', style={'color': PALETTE['text'],
                                              'marginBottom': '12px'}),
        table])],
        style={'background': PALETTE['card'],
               'border': f'1px solid {PALETTE["border"]}',
               'borderRadius': '8px'})

    status = f'Dernier vol : {last.strftime("%H:%M:%S")}'
    return kpis, card, status


if __name__ == '__main__':
    print('Dashboard DST Airlines : http://0.0.0.0:8050')


if __name__ == '__main__':
    print('Dashboard DST Airlines : http://0.0.0.0:8050')
    app.run(debug=False, host='0.0.0.0', port=8050)
