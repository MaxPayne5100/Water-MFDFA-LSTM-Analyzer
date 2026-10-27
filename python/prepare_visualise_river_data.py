import pandas as pd
import os
import logging
from utils import load_config, project_path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from deep_translator import GoogleTranslator
from data_quality import validate_time_series_file
from lineage import publish_atomic_output, temporary_output_path


logger = logging.getLogger(__name__)


def prepare_river_data(config_file_path: str) -> tuple[pd.DataFrame, str]:
    """
    Analyse prepared river data and prepare it for machine learning processing.

    Args:
        config_file_path (str): Path to config file.

    Returns:
        pd.DataFrame:: Prepared dataframe with data and indicator data for machine learning processing.
        str: River name.
    """
    temporary_ts_path = None
    try:
        config_params = load_config(config_file_path)
        bronze_folder = project_path(config_params['BRONZE_FOLDER'])
        curated_folder = project_path(config_params['CURATED_FOLDER'])
        os.makedirs(curated_folder, exist_ok=True)
        indicator = config_params.get('INDICATOR', {})
        filter_data_path = os.path.join(bronze_folder, 'combined_filtered_data.csv')
        river_name_column = config_params.get('RIVER_NAME_COLUMN', {})
        river_name_eng = config_params.get('RIVER_NAME_VALUE_ENG', {})

        filtered_data = pd.read_csv(filter_data_path)

        required_columns = {'Controle_Date', river_name_column, indicator}
        missing_columns = required_columns.difference(filtered_data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in filtered data: {sorted(missing_columns)}"
            )

        duplicate_count = int(filtered_data.duplicated().sum())
        if duplicate_count:
            raise ValueError(f"Filtered data contains {duplicate_count} duplicate rows")

        # Monitoring datasets legitimately have NULLs in optional indicators. Only
        # the configured indicator and fields needed to build the time series are
        # required for this preparation step.
        filtered_data = filtered_data.dropna(
            subset=[indicator, 'Controle_Date', river_name_column]
        ).copy()
        if filtered_data.empty:
            raise ValueError("No valid rows remain after required-field validation")

        # change date values from string to datetime object
        date_formats = [
            '%Y-%m-%d',
            '%d.%m.%Y'
        ]

        filtered_data['_Raw_Date_String'] = filtered_data['Controle_Date'].astype(str)

        filtered_data['Controle_Date'] = pd.to_datetime(
            filtered_data['_Raw_Date_String'],
            format=date_formats[0],
            errors='coerce'
        )

        for fmt in date_formats[1:]:
            mask = filtered_data['Controle_Date'].isna()

            if not mask.any():
                break

            raw_data_to_parse = filtered_data.loc[mask, '_Raw_Date_String']

            parsed_subset = pd.to_datetime(raw_data_to_parse, format=fmt, errors='coerce')

            filtered_data.loc[mask, 'Controle_Date'] = parsed_subset

        filtered_data.drop(columns=['_Raw_Date_String'], inplace=True)

        filtered_data = filtered_data.rename(columns={'Controle_Date': 'date'})
        filtered_data = filtered_data.sort_values('date', ascending=True)
        filtered_data.reset_index(drop=True, inplace=True)
        filtered_data.set_index('date', inplace = True)

        river = filtered_data[river_name_column].iloc[0]
        indicator_data = filtered_data[[f'{indicator}']]
        indicator_data = indicator_data.dropna()

        ts_data_path = os.path.join(curated_folder, f'{river_name_eng}_{indicator}_ts.csv')
        temporary_ts_path = temporary_output_path(ts_data_path)
        indicator_data.to_csv(temporary_ts_path, index=True, encoding='utf-8')
        report = validate_time_series_file(temporary_ts_path, indicator, config_params['HORIZON'])
        lineage_path = publish_atomic_output(
            temporary_ts_path,
            ts_data_path,
            config_file_path,
            "curated",
            input_files=[filter_data_path],
            extra={"row_count": report["rows"], "indicator": indicator},
        )
        print(f"Prepared time-series quality passed: {report}")
        print(f"Lineage metadata written to: {lineage_path}")
        return (indicator_data, river)
    except Exception:
        logger.exception("Failed to prepare river data")
        raise
    finally:
        if temporary_ts_path is not None:
            temporary_ts_path.unlink(missing_ok=True)


def generate_dashboard(config_file_path: str, indicator_river_data: pd.DataFrame, river: str):
    """
    Generate dashboard with general statistics for the time series data that we pass as input parameter.

    Args:
        config_file_path (str): Path to config file.
        indicator_river_data (pd.DataFrame): Time series data to analyse and visualize
    """
    temporary_png_path = None
    try:
        config_params = load_config(config_file_path)
        ts_analysis_data = project_path(config_params['TS_ANALYSIS_FOLDER'])
        os.makedirs(ts_analysis_data, exist_ok=True)
        indicator = config_params.get('INDICATOR', {})

        decomposed_indicator = sm.tsa.seasonal_decompose(indicator_river_data, period=len(indicator_river_data)//2, model='additive')

        plt.subplots_adjust(wspace=0, hspace=1)
        fig = plt.figure(figsize=(25, 20))
        gs = fig.add_gridspec(nrows=4, ncols=3, height_ratios=[1, 1, 1, 1], width_ratios=[1, 0.7, 0.7])

        fig.suptitle(f'Дешборд аналізу часового ряду {indicator} даних річки {river}', fontsize=18, fontweight='bold', y=0.98)

        plt.tight_layout(rect=(1, 1, 1, 1)) # [left, bottom, right, top] margins

        ax_observed = fig.add_subplot(gs[0, 0]) # First row, first column
        decomposed_indicator.observed.plot(ax=ax_observed, color='black', label=f'{indicator}')
        ax_observed.set_title(f'Часовий ряд {indicator}', fontsize=14)
        ax_observed.set_ylabel(f'{indicator}', fontsize=12)
        ax_observed.set_xlabel('')
        ax_observed.grid(True)
        ax_observed.set_facecolor('#f9f9f9')

        ax_trend = fig.add_subplot(gs[1, 0], sharex=ax_observed)
        decomposed_indicator.trend.plot(ax=ax_trend, color='black', label='Trend')
        ax_trend.set_title('Тренд', fontsize=14)
        ax_trend.set_ylabel(f'{indicator}', fontsize=12)
        ax_trend.set_xlabel('')
        ax_trend.grid(True)
        ax_trend.set_facecolor('#f9f9f9')

        ax_seasonal = fig.add_subplot(gs[2, 0], sharex=ax_observed)
        decomposed_indicator.seasonal.plot(ax=ax_seasonal, color='black', label='Seasonal')
        ax_seasonal.set_title('Сезонність', fontsize=14)
        ax_seasonal.set_ylabel(f'{indicator}', fontsize=12)
        ax_seasonal.set_xlabel('')
        ax_seasonal.grid(True)
        ax_seasonal.set_facecolor('#f9f9f9')

        ax_residuals = fig.add_subplot(gs[3, 0], sharex=ax_observed)
        decomposed_indicator.resid.plot(ax=ax_residuals, color='black', label='Residuals')
        ax_residuals.set_title('Надлишки', fontsize=14)
        ax_residuals.set_ylabel(f'{indicator}', fontsize=12)
        ax_residuals.set_xlabel('Дата', fontsize=12)
        ax_residuals.grid(True)
        ax_residuals.set_facecolor('#f9f9f9')

        ax_hist = fig.add_subplot(gs[2:4, 1:2]) # Spans rows 0 and 1, column 1 (of the right two columns)
        indicator_river_data.hist(ax=ax_hist, bins=25, color="black", edgecolor='black', alpha=0.8)
        ax_hist.set_xlabel(f"{indicator}", fontsize=12)
        ax_hist.set_ylabel("Частота", fontsize=12)
        ax_hist.set_title(f"Розподіл {indicator}", fontsize=14)
        ax_hist.set_facecolor('#f9f9f9')
        ax_hist.grid(axis='y', linestyle=':', alpha=0.7)

        descriptive_stats = indicator_river_data.describe()
        ax_stats = fig.add_subplot(gs[2:4, 2:3])
        descriptive_stats.loc[['mean', 'std']].T.plot(ax=ax_stats, kind='bar', yerr='std', capsize=4)
        ax_stats.set_title(f'Середнє та стандартна девіація для {indicator}')
        ax_stats.set_ylabel('Значення')
        ax_stats.grid(axis='y', linestyle='--', alpha=0.7)

        ax_acf = fig.add_subplot(gs[0, 1:3]) # Spans rows 0 and 1, column 2 (of the right two columns)
        sm.graphics.tsa.plot_acf(indicator_river_data.values, lags=100, ax=ax_acf, alpha=0.05)
        ax_acf.set_title('Автокореляція (ACF)', fontsize=14)
        ax_acf.set_ylabel('Кореляція', fontsize=12)
        ax_acf.set_xlabel('Лаг', fontsize=12)
        ax_acf.grid(True)
        ax_acf.set_facecolor('#f9f9f9')

        if ax_acf.lines:
            ax_acf.lines[0].set_color('black')
            for line in ax_acf.lines[1:]:
                line.set_color('black')
                line.set_alpha(0.3)

        ax_pacf = fig.add_subplot(gs[1, 1:3]) # Spans rows 2 and 3, and columns 1 and 2 (of the right two columns)
        sm.graphics.tsa.plot_pacf(indicator_river_data.values, lags=100, ax=ax_pacf, alpha=0.05)
        ax_pacf.set_title('Часткова Автокореляція (PACF)', fontsize=14)
        ax_pacf.set_ylabel('Кореляція', fontsize=12)
        ax_pacf.set_xlabel('Лаг', fontsize=12)
        ax_pacf.grid(True)
        ax_pacf.set_facecolor('#f9f9f9')

        if ax_pacf.lines:
            ax_pacf.lines[0].set_color('black')
            for line in ax_pacf.lines[1:]:
                line.set_color('black')
                line.set_alpha(0.3)

        output_filename_png = os.path.join(ts_analysis_data, f'{river}_{indicator}_часовий_ряд_дешборд.png')

        try:
                temporary_png_path = temporary_output_path(output_filename_png)
                plt.savefig(
                    temporary_png_path,
                    format='png',
                    dpi=300,
                    bbox_inches='tight',
                )
                curated_input_path = os.path.join(
                    project_path(config_params['CURATED_FOLDER']),
                    f"{config_params['RIVER_NAME_VALUE_ENG']}_{indicator}_ts.csv",
                )
                lineage_path = publish_atomic_output(
                    temporary_png_path,
                    output_filename_png,
                    config_file_path,
                    "analysis",
                    input_files=[curated_input_path],
                    extra={"artifact_type": "dashboard"},
                )
                print(f"Dashboard saved successfully as {output_filename_png}")
                print(f"Lineage metadata written to: {lineage_path}")
                print()
                plt.close(fig)
        except Exception:
                logger.exception("Failed to save river dashboard")
                raise
    except Exception:
        logger.exception("Failed to generate river dashboard")
        raise
    finally:
        if temporary_png_path is not None:
            temporary_png_path.unlink(missing_ok=True)


if __name__ == "__main__":
    config_file_path = project_path("data_lake", "config.json")
    indicator_data, river = prepare_river_data(config_file_path)
    generate_dashboard(config_file_path, indicator_data, river)
