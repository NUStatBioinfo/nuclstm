from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from flask_bootstrap import Bootstrap
from flask_caching import Cache
from nuc_viewer_app.forms import UploadForm
from nuc_viewer_app.config import Config
from nuc_viewer_app.plotting import *
from nuc_viewer_app.analysis import *
from pandas.io.json import read_json
from pandas import read_csv
from bokeh.embed import components
import re


# create and configure application.
app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config.from_object(Config)

# initialize Session, Cache.
sess = Session()
sess.init_app(app)
cache = Cache()
cache.init_app(app)

# fields that will expand y-axis in base-index plot.
expand_y_fields = ['nuclstm_preds_sum', 'ncp_occup_score']


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()

    if request.method == 'POST' and form.validate_on_submit():
        selected_file = request.files['input_file']

        # determine chromosome number being analyzed.
        chrom = re.findall(r'\d+', selected_file.filename)[-1]
        if not chrom:
            latins = get_latin_chrom()
            for roman in latins.keys():
                if re.search(roman, selected_file.filename):
                    chrom = latins[roman]

        session['chromosome'] = str(chrom)

        df = read_csv(selected_file)

        # add desired features if they're not present.
        for col in expand_y_fields:
            if col not in df.columns:
                if col == 'nuclstm_preds_sum':
                    df[col] = df['nuclstm_preds'].rolling(147
                                                          , min_periods=1).sum()
                elif col == 'ncp_occup_score':
                    df[col] = df['NCP'].rolling(147
                                                , min_periods=1).sum()

        # run k-sensitivity analysis.
        session['k_sensitivity'] = get_ksensitivity(df).to_json()

        # summarize distance from chemical map and store it in session.
        summary_df = DataFrame({'NuPoP total NCP': df['nupop_ncp'].sum()
                                , 'Nuclstm total NCP': df['nuclstm_ncp'].sum()
                                , 'NuPoP mean c2c distance': np.mean(get_minimum_binary_distance(np.where(df['nupop_ncp'] == 1)[0]
                                                                                                 ,y=np.where(df['nucleosome'] == 1)[0]))
                                , 'Nuclstm mean c2c distance': np.mean(get_minimum_binary_distance(np.where(df['nuclstm_ncp'] == 1)[0]
                                                                                                 , y=np.where(df['nucleosome'] == 1)[0]))
                                , 'Chemical map nucleosomes': df['nucleosome'].sum()}
                               , index=[0])

        session['summary'] = summary_df.to_json()

        # store data in session.
        session['df'] = df.to_json()

        return redirect(url_for('uploaded_file'))
    else:
        return render_template('index.html'
                               , form=form)


@app.route('/uploaded_file', methods=['GET', 'POST'])
# @cache.cached(timeout=300)
def uploaded_file():
    df = read_json(session['df'])
    df.sort_values('pos'
                   , inplace=True)

    # construct Bokeh plot of k-distance sensitivity and obtain js scripting.
    ksens_df = read_json(session['k_sensitivity'])
    ksens_df.sort_values('k'
                         , inplace=True)
    ksens = bokeh_lines(ksens_df['k'].values.tolist()
                        , ys=[ksens_df[x].values.tolist() for x in ['nupop_tpr', 'nuclstm_tpr']]
                        , labels=['nupop', 'nuclstm']
                        , x_label='distance (bp)'
                        , y_label='sensitivity (%)')

    ksens_script, ksens_div = components(ksens)

    # get model comparison summary
    summary_df = read_json(session['summary'])

    # keep and sort the features we'd be interested in viewing.
    feature_names = list(set(df.columns.tolist()) - set(['Chr', 'pos', 'seq']))
    feature_names.sort()

    if request.method == 'POST':

        # obtain desired features to analyze
        selected_features = request.form.getlist('feature_names')

        if selected_features:
            selected_features.sort()

            # determine max of y-axis
            try:
                y_max = float(request.form['y_max'])
            except:
                y_max = max(df[selected_features].max())

            # get start/end positions of chromosome to load (only seq_len will be displayed at once)
            start = max(int(request.form['start_position']), 0)
            end = int(request.form['end_position'])

            # construct Bokeh base position plot and obtain js scripting to send to ui.
            pos = base_position_plot(df
                                     , features=selected_features
                                     , start=start
                                     , end=end
                                     , y_max=y_max
                                     , seq_len=1000
                                     , plot_width=900
                                     , plot_height=600)

            pos_script, pos_div = components(pos)

            # obtain correlation matrix of selected features.
            corr = df[selected_features].corr()

            print('k-sensitivity script:\n{0}'.format(ksens_script))
            print('k-sensitivity div:\n{0}'.format(ksens_div))

            return render_template('chromosome.html'
                                   , chromosome=session['chromosome']
                                   , feature_names=feature_names
                                   , selected_features=selected_features
                                   , summary_table=summary_df.to_html(index=False)
                                   , pos_script=pos_script
                                   , pos_div=pos_div
                                   , ksens_script=ksens_script
                                   , ksens_div=ksens_div
                                   , corr_table=corr.to_html())

    return render_template('chromosome.html'
                           , chromosome=session['chromosome']
                           , feature_names=feature_names
                           , selected_features=None
                           , summary_table=summary_df.to_html(index=False)
                           , ksens_script=ksens_script
                           , ksens_div=ksens_div)