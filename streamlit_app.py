import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
# from scipy.optimize import curve_fit 
from plotly.subplots import make_subplots
import functools
from companies import COMPANY_GROUPS, COMPANIES

with open("load-mathjax.js", "r") as f:
    js = f.read()
    st.components.v1.html(f"<script>{js}</script>", height=0)

COMPANY_COL = 'Company'
LEVEL_INDEX_COL = 'Level Index'
LEVEL_COL = 'Level Title'
COMP_TC_COL = 'Total Compensation'
COMP_SALARY_COL = 'Salary'
COMP_STOCK_COL = 'Stock'
COMP_BONUS_COL = 'Bonus'
YOE_COL = 'Years of Experience'
OFFER_DATE_COL = 'Offer Date'
FOCUS_COL = 'Focus'
N_OFFERS_COL = '# Offers'

@st.cache_data
def data():
    output_dir = Path('output')
    jsons = list(output_dir.glob("*.json"))
    dfs = []
    for file_path in jsons:
        as_dict = json.load(open(file_path))
        levels = as_dict['props']['pageProps'].get('averages', None)
        if not levels:
            continue
        for level in levels:
            df = pd.DataFrame.from_records(level['samples'])
            for k, v in level.items():
                if k == 'samples':
                    continue
                df[f"level_attr_{k}"] = v
            dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    del dfs

    df = df.rename(columns={
        'level_attr_levelIndex': LEVEL_INDEX_COL,
        'totalCompensation': COMP_TC_COL,
        'company': COMPANY_COL,
        'yearsOfExperience': YOE_COL,
        'level': LEVEL_COL,
        'offerDate': OFFER_DATE_COL,
        'baseSalary': COMP_SALARY_COL,
        'avgAnnualStockGrantValue': COMP_STOCK_COL,
        'avgAnnualBonusValue': COMP_BONUS_COL,
        'focusTag': FOCUS_COL,
    })

    return df

DEFAULT_COMPANIES = COMPANY_GROUPS['Market Leaders']

class StreamlitQueryParam:
    def __init__(self, name, default, to_clear=None):
        self.name = name
        self.default = default
        self.to_clear = to_clear or []
    
    def input_cast(self, value):
        return value

    def __enter__(self):
        if self.name in st.query_params:
            self.value = self.input_cast(st.query_params[self.name])
        else:
            self.value = self.default
        self.initial_value = self.value
        return self
    
    def output_cast(self, value):
        return value
    
    @property
    def has_changed(self):
        return self.value != self.initial_value
    
    @property
    def is_default(self):
        return self.value == self.default
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.has_changed:
            if not self.is_default:
                st.query_params[self.name] = self.output_cast(self.value)
            elif self.name in st.query_params:
                del st.query_params[self.name]
            
            for to_clear in self.to_clear:
                if to_clear.name in st.query_params:
                    del st.query_params[to_clear.name]

class StreamlitQueryParamJSON(StreamlitQueryParam):
    def input_cast(self, value):
        return json.loads(value)

    def output_cast(self, value):
        return json.dumps(value)

class StreamlitQueryParamNumber(StreamlitQueryParam):
    def __init__(self, name, default, to_clear=None, type=float):
        super().__init__(name, default, to_clear)
        self.type = type

    def input_cast(self, value):
        return self.type(value)

    def output_cast(self, value):
        return str(value)

class StreamlitQueryParamBoolean(StreamlitQueryParamNumber):
    def __init__(self, name, default, to_clear=None):
        super().__init__(name, default, to_clear, type=bool)

    def input_cast(self, value):
        return super().input_cast(int(value))

    def output_cast(self, value):
        return super().output_cast(int(value))


class StreamlitApp():
    def __init__(self):
        self._raw_df = data()
    
    @property
    def df(self):
        return self._raw_df.copy()
    
    DEFAULT_FOCUS_AREA = ''
    @functools.cached_property
    def focus_area(self):
        with StreamlitQueryParam('focus_area', self.DEFAULT_FOCUS_AREA) as focus_area_qp:
            focus_area_qp.value = st.text_input("Focus Area", value=st.query_params.get('focus_area', self.DEFAULT_FOCUS_AREA), key='focus_area')
            return focus_area_qp.value

    @property
    def max_yac(self):
        return int(self._raw_df['yearsAtCompany'].max())
    
    DEFAULT_YAC_MIN = 0
    DEFAULT_YAC_MAX = None
    @functools.cached_property
    def yac(self):
        with (
            StreamlitQueryParamNumber('yac_min', self.DEFAULT_YAC_MIN, type=int) as yac_min_qp,
            StreamlitQueryParamNumber('yac_max', self.DEFAULT_YAC_MAX or self.max_yac, type=int) as yac_max_qp,
        ):
            yac_min_qp.value, yac_max_qp.value = st.slider(
                "Years at Company",
                value=(yac_min_qp.value, yac_max_qp.value),
                min_value=0,
                max_value=self.max_yac,
                step=1,
                format="%d",
            )
            return (yac_min_qp.value, yac_max_qp.value)
    
    @property
    def max_level_index(self):
        return int(self._raw_df[LEVEL_INDEX_COL].max())
    
    DEFAULT_LEVEL_INDEX_MIN = 0
    DEFAULT_LEVEL_INDEX_MAX = None
    @functools.cached_property
    def level_index(self):
        with (
            StreamlitQueryParamNumber('level_index_min', self.DEFAULT_LEVEL_INDEX_MIN, type=int) as level_index_min_qp,
            StreamlitQueryParamNumber('level_index_max', self.DEFAULT_LEVEL_INDEX_MAX or self.max_level_index, type=int) as level_index_max_qp,
        ):
            level_index_min_qp.value, level_index_max_qp.value = st.slider(
                "Level Index",
                value=(level_index_min_qp.value, level_index_max_qp.value),
                min_value=0,
                max_value=self.max_level_index,
                step=1,
                format="%d",
            )
            return (level_index_min_qp.value, level_index_max_qp.value)
    
    @property
    def unique_companies(self):
        return COMPANIES
        # return self.df[COMPANY_COL].unique()

    DEFAULT_INDIVIDUAL_COMPANIES = False
    DEFAULT_COMPANY_GROUPS = ['Market Leaders']
    QUERY_PARAM_COMPANY_GROUPS = 'company_groups'
    QUERY_PARAM_COMPANIES = 'companies'
    QUERY_PARAM_INDIVIDUAL_COMPANIES = 'individual_companies'

    @functools.cached_property
    def companies(self):
        with StreamlitQueryParamJSON(self.QUERY_PARAM_COMPANY_GROUPS, self.DEFAULT_COMPANY_GROUPS) as company_groups_qp:
            company_group_companies = sum((COMPANY_GROUPS[group] for group in company_groups_qp.value), [])
            company_group_companies = sorted(list(set(company_group_companies)))
            company_group_companies = company_group_companies or self.unique_companies
            with (
                StreamlitQueryParamJSON(self.QUERY_PARAM_COMPANIES, company_group_companies) as companies_qp,
                StreamlitQueryParamBoolean(self.QUERY_PARAM_INDIVIDUAL_COMPANIES, self.DEFAULT_INDIVIDUAL_COMPANIES) as individual_companies_qp,
            ):
                individual_companies_qp.value = st.checkbox("Choose Individual Companies", value=individual_companies_qp.value)
                company_groups_qp.value = st.multiselect(
                    "Company Groups",
                    options=COMPANY_GROUPS.keys(),
                    default=company_groups_qp.value,
                    disabled=individual_companies_qp.value,
                )

                if not individual_companies_qp.value:
                    companies_qp.value = company_group_companies
                companies_qp.value = sorted(list(st.multiselect(
                    "Companies",
                    options=self.unique_companies,
                    default=sorted(list(companies_qp.value)),
                    disabled=not individual_companies_qp.value,
                    key=self.QUERY_PARAM_COMPANIES,
                )))
                return companies_qp.value
    
    DEFAULT_FILTERED_DMAS = [807]
    DMAs = {
        807: "Greater SF Bay",
        819: "Greater Seattle",
        501: "Greater NYC",
        803: "Greater LA",
    }
    @functools.cached_property
    def filtered_dmas(self):
        with StreamlitQueryParamJSON('filtered_dmas', self.DEFAULT_FILTERED_DMAS) as filtered_dmas_qp:
            filtered_dmas_qp.value = st.multiselect(
                "Filter to Regions",
                options=self.DMAs.keys(),
                format_func=lambda x: self.DMAs[x],
                default=filtered_dmas_qp.value,
            )
            return filtered_dmas_qp.value
        
    
    DEFAULT_FILTER_TO_US = True
    @functools.cached_property
    def filter_to_us(self):
        with StreamlitQueryParamBoolean('filter_to_us', self.DEFAULT_FILTER_TO_US) as input_qp:
            input_qp.value = st.checkbox(
                "Filter to US",
                value=input_qp.value,
            )
            return input_qp.value
    
    DEFAULT_YOE = 8
    @functools.cached_property
    def yoe(self):
        with StreamlitQueryParamNumber('yoe', self.DEFAULT_YOE, type=int) as yoe_qp:
            yoe_qp.value = st.number_input(
                "Years of Experience",
                value=yoe_qp.value,
                min_value=0,
                max_value=10,
            )
            return yoe_qp.value
    
    DEFAULT_LOG_Y = False
    @functools.cached_property
    def plot_options(self):
        with StreamlitQueryParamBoolean('log_y', self.DEFAULT_LOG_Y) as log_y_qp:
            with st.expander("Plot Options", expanded=log_y_qp.value):
                log_y_qp.value = st.checkbox(
                    "Log Y", value=log_y_qp.value,
                )
            return (log_y_qp.value, )
    
    @property
    def log_y(self):
        return self.plot_options[0]
    
    @property
    def filtered_df(self):
        df = self.df
        
        if self.focus_area:
            df = df[df[FOCUS_COL].str.contains(self.focus_area, na=False)]
        
        yac_min, yac_max = self.yac
        df = df[df['yearsAtCompany'] >= yac_min]
        df = df[df['yearsAtCompany'] <= yac_max]
        
        level_index_min, level_index_max = self.level_index
        df = df[df[LEVEL_INDEX_COL] >= level_index_min]
        df = df[df[LEVEL_INDEX_COL] <= level_index_max]
        
        if self.companies:
            df = df[df[COMPANY_COL].isin(self.companies)]
        
        if self.filtered_dmas:
            df = df[df['dmaId'].isin(self.filtered_dmas)]
        
        if self.filter_to_us:
            df = df[df['countryId'] == 254]
        
        return df
    
    def model_func(self, x, a, b):
        return np.exp(b + a * x)
    
    def exp_fit(self, x, y):
        least_squares_line = np.polyfit(x, np.log(y), 1)
        linspace = np.linspace(x.min(), x.max(), 100)
        predicted = self.model_func(linspace, *least_squares_line)
        return {
            'x': linspace,
            'y': predicted, 
            'least_squares_line': least_squares_line,
        }
    
    def violin_plot_trace(self, df, y_col):
        return go.Violin(
            y=df[y_col],
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
            name='Overall',
        )
    
    def scatter_plot_trace(self, df, x_col, y_col, color_col):
        hover_template = "<br>".join(f"{k}: %{{customdata[{i}]{':,.2f' if k in [COMP_TC_COL, COMP_SALARY_COL, COMP_STOCK_COL, COMP_BONUS_COL] else ''}}}" for i, k in enumerate(df.columns))
        return go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            name=y_col,
            customdata=df,
            hovertemplate=hover_template,
            marker=dict(
                color=df[color_col],
            ),
        )
    
    def trendline_trace(self, trendline_data):
        least_squares_line = trendline_data['least_squares_line']
        latex_text = f"$y = {np.exp(least_squares_line[1]):,.2f}e^{{{least_squares_line[0]:.3f}x}}$"
        return go.Scatter(
                x=trendline_data['x'],
                y=trendline_data['y'],
                mode='lines',
                name=latex_text,#'Trendline',
        )
    
    def expected_trace(self, trendline_data, yoe):
        return go.Scatter(
            x=[yoe],
            y=[self.model_func(yoe, *trendline_data['least_squares_line'])],
            name=f'Expected',
            mode='markers',
            marker=dict(size=10)  # Marker size in pixels
        )
    
    def scatter_plot(
        self,
        df,
        x_col,
        y_col,
        color_col,
        log_y,
        yoe,
        x_title=None,
        y_title=None,
        title=None,
        subtitle=" "
    ):
        left, right = 1, 5
        fig = make_subplots(
            rows=1, cols=left + right,
            shared_yaxes=True,
            specs=[
                [{"colspan": left}, {"colspan": right}] + [{},] * (right - left),
            ],
        )
        fig.add_trace(
            self.violin_plot_trace(df, y_col),
            row=1, col=1,
        )
        trendline_data = self.exp_fit(df[x_col], df[y_col])
        fig.add_traces(
            [
                self.scatter_plot_trace(df, x_col, y_col, color_col),
                self.trendline_trace(trendline_data),
                self.expected_trace(trendline_data, yoe),
            ],
            rows=1, cols=left + 1,
        )
        if not x_title:
            x_title = f"{self.yoe_agg_func_title} {x_col}"
        fig.update_xaxes(title_text=x_title, row=1, col=left + 1)
        if not y_title:
            y_title = f"{self.tc_agg_func_title} {y_col}"
        fig.update_yaxes(title_text=y_title, row=1, col=left)
        if not title:
            title = f"{y_title} over {x_title}"
        expected_text = f"Expected: ${self.model_func(yoe, *trendline_data['least_squares_line']):,.2f} [@ {yoe} {YOE_COL}]<br>N: {len(df)}"
        fig.update_layout(title_text=f"{title}<br><sup>{subtitle} [N: {len(df)}]<br>{expected_text}<br>&nbsp;</sup>")
        if log_y:
            fig.update_yaxes(type="log")
        return fig
    
    @functools.cached_property
    def tc_by_offer(self):
        return self.scatter_plot(
            df=self.filtered_df[[
                COMPANY_COL,
                LEVEL_COL,
                LEVEL_INDEX_COL,
                FOCUS_COL,

                COMP_TC_COL,
                COMP_SALARY_COL,
                COMP_STOCK_COL,
                COMP_BONUS_COL,
                OFFER_DATE_COL,
                
                YOE_COL,
            ]],
            x_col=YOE_COL,
            y_col=COMP_TC_COL,
            color_col=LEVEL_INDEX_COL,
            log_y=self.log_y,
            yoe=self.yoe,
            x_title=YOE_COL,
            y_title=COMP_TC_COL,
            subtitle="By Individual Offer"
        )
    
    DEFAULT_USE_YOE_MEAN = True
    DEFAULT_YOE_QUANTILE = 0.5
    @functools.cached_property
    def yoe_options(self):
        with (
            StreamlitQueryParamNumber('yoe_quantile', self.DEFAULT_YOE_QUANTILE) as yoe_quantile_qp,
            StreamlitQueryParamBoolean('use_yoe_mean', self.DEFAULT_USE_YOE_MEAN, to_clear=[yoe_quantile_qp]) as use_yoe_mean_qp
        ):
            with st.expander(f"{YOE_COL} Options", expanded=not use_yoe_mean_qp.value):
                use_yoe_mean_qp.value = st.checkbox("Use Mean Years of Experience", value=use_yoe_mean_qp.value)
                if use_yoe_mean_qp.value:
                    yoe_agg_func = 'mean'
                else:
                    yoe_quantile_qp.value = st.number_input(
                        "YOE Quantile",
                        value=yoe_quantile_qp.value,
                        min_value=0.0,
                        max_value=1.0,
                    )
                    yoe_agg_func = lambda s: s.quantile(yoe_quantile_qp.value, interpolation='higher')
            return yoe_agg_func, yoe_quantile_qp.value
    
    @property
    def yoe_agg_func(self):
        return self.yoe_options[0]
    
    @property
    def yoe_quantile(self):
        return self.yoe_options[1]
    
    def mean_or_percentile(self, fn, quantile):
        if fn == 'mean':
            return "Mean"
        else:
            return f"{quantile * 100 :.0f}th Percentile"
    
    @property
    def yoe_agg_func_title(self):
        return self.mean_or_percentile(self.yoe_agg_func, self.yoe_quantile)
    
    DEFAULT_USE_TC_MEAN = True
    DEFAULT_TC_QUANTILE = 0.75
    @functools.cached_property
    def tc_options(self):
        with (
            StreamlitQueryParamNumber('tc_quantile', self.DEFAULT_TC_QUANTILE) as tc_quantile_qp,
            StreamlitQueryParamBoolean('use_tc_mean', self.DEFAULT_USE_TC_MEAN, to_clear=[tc_quantile_qp]) as use_tc_mean_qp,
        ):
            with st.expander(f"{COMP_TC_COL} Aggregation Options", expanded=not use_tc_mean_qp.value):
                use_tc_mean_qp.value = st.checkbox("Use Mean Total Compensation", value=use_tc_mean_qp.value)
                if use_tc_mean_qp.value:
                    tc_agg_func = 'mean'
                else:
                    tc_quantile_qp.value = st.number_input(
                        "TC Quantile",
                        value=tc_quantile_qp.value,
                        min_value=0.0,
                        max_value=1.0,
                    )
                    tc_agg_func = lambda s: s.quantile(tc_quantile_qp.value, interpolation='higher')
            return tc_agg_func, tc_quantile_qp.value
    
    @property
    def tc_agg_func(self):
        return self.tc_options[0]
    
    @property
    def tc_quantile(self):
        return self.tc_options[1]
    
    @property
    def tc_agg_func_title(self):
        return self.mean_or_percentile(self.tc_agg_func, self.tc_quantile)
    
    @property
    def lt_yoe_col(self):
        return f'LT {self.yoe} YOE'
    
    @functools.cached_property
    def aggregated_by_level_df(self):
        to_agg = self.filtered_df
        to_agg[self.lt_yoe_col] = (to_agg[YOE_COL] <= self.yoe).astype(int)
        grouped_by_level = to_agg.groupby([COMPANY_COL, LEVEL_COL, LEVEL_INDEX_COL])
        level_by_yoe = grouped_by_level.agg({
            YOE_COL: self.yoe_agg_func,
            COMP_TC_COL: self.tc_agg_func,
            FOCUS_COL: 'count',
            self.lt_yoe_col: 'sum',
        }).reset_index()
        level_by_yoe = level_by_yoe.rename(columns={FOCUS_COL: N_OFFERS_COL})
        return level_by_yoe
        
    
    @functools.cached_property
    def aggregated_by_level(self):
        return self.scatter_plot(
            df=self.aggregated_by_level_df,
            x_col=YOE_COL,
            y_col=COMP_TC_COL,
            color_col=LEVEL_INDEX_COL,
            log_y=self.log_y,
            yoe=self.yoe,
            subtitle="Aggregated By Level"
        )
    
    @functools.cached_property
    def total_compensation_by_yoe(self):
        total_compensation_by_yoe = self.filtered_df.groupby(YOE_COL)[COMP_TC_COL].agg(**{
            COMP_TC_COL: self.tc_agg_func,
            'Count': 'count'
        }).reset_index()
        total_compensation_by_yoe = total_compensation_by_yoe[total_compensation_by_yoe['Count'] > 2]
        return total_compensation_by_yoe
    
    @functools.cached_property
    def aggregated_by_yoe(self):
        return self.scatter_plot(
            df=self.total_compensation_by_yoe,
            x_col=YOE_COL,
            y_col=COMP_TC_COL,
            color_col=YOE_COL,
            log_y=self.log_y,
            yoe=self.yoe,
            subtitle="Aggregated By Years of Experience"
        )
    
    @functools.cached_property
    def best_levels(self):
        best_levels = self.aggregated_by_level_df
        best_levels = best_levels[
            self.aggregated_by_level_df[self.lt_yoe_col] > 0
        ]
        best_levels = best_levels.sort_values(COMP_TC_COL, ascending=False)
        best_levels = best_levels.reset_index(drop=True)
        best_levels = best_levels[[
            COMPANY_COL,
            LEVEL_COL,
            LEVEL_INDEX_COL,
            COMP_TC_COL,
            YOE_COL,
            self.lt_yoe_col,
            N_OFFERS_COL,
        ]]
        best_levels = best_levels.rename(columns={self.lt_yoe_col: f'# Offers w/ {self.lt_yoe_col}'})
        return best_levels
    
    @functools.cached_property
    def best_levels_plot(self):
        fig = px.scatter(
            data_frame=self.best_levels,
            x=YOE_COL,
            y=COMP_TC_COL,
            hover_data=self.best_levels.columns,
            color=COMPANY_COL,
            size=LEVEL_INDEX_COL,
            log_y=self.log_y,
        )
        fig.update_layout(title_text="Best Levels")
        return fig
    
    FILTER_PARAMS = set([
        'focus_area',
        'yac_min',
        'yac_max',
        'level_index_min',
        'level_index_max',
        QUERY_PARAM_COMPANY_GROUPS,
        QUERY_PARAM_COMPANIES,
        QUERY_PARAM_INDIVIDUAL_COMPANIES,
        'filtered_dmas',
        'filter_to_us',
    ])
    @functools.cached_property
    def data_filters(self):
        filters_set = st.query_params.keys() & self.FILTER_PARAMS
        with st.expander("Data Filters", expanded=bool(filters_set)):
            _ = self.focus_area
            _ = self.yac
            _ = self.level_index
            _ = self.companies
            _ = self.filtered_dmas
            _ = self.filter_to_us
    
    @functools.cached_property
    def candidate_inputs(self):
        with st.expander("Candidate Inputs", expanded=True):
            _ = self.yoe

    # @st.fragment(run_every=1)
    def display_plots(self):
        tc_by_offer_tab, aggregated_tab, best_levels_tab = st.tabs(["By Offer", "Aggregated", "Best Levels"])
        with tc_by_offer_tab:
            st.write(self.tc_by_offer)
        with aggregated_tab:
            _ = self.yoe_options
            _ = self.tc_options
            by_level_tab, by_yoe_tab = st.tabs(["By Level", "By Years of Experience"])
            with by_level_tab:
                st.write(self.aggregated_by_level)
            with by_yoe_tab:
                st.write(self.aggregated_by_yoe)
        with best_levels_tab:
            st.write(self.best_levels_plot)
            st.write(self.best_levels)

    def main(self):
        _ = self.data_filters
        _ = self.candidate_inputs
        _ = self.plot_options
        st.markdown("---")

        self.display_plots()

    


if __name__ == "__main__":
    app = StreamlitApp()
    app.main()
