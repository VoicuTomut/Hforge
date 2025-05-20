from prettytable import PrettyTable

def main():
    table = PrettyTable(["Type", "Last model save", "New prediction"])
    print(table)

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Read data
    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "palashfendarkar/wa-fnusec-telcocustomerchurn",
      file_path,
      # Provide any additional arguments like
      # sql_query or pandas_kwargs. See the
      # documenation for more information:
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    # Define subplot titles and data columns
    data_cols = ['PhoneService' ,'MultipleLines' ,'InternetService' ,'OnlineBackup' ,'DeviceProtection' ,'TechSupport' ,'StreamingTV' ,'StreamingMovies']
    titles = ['Phone Service' ,'Multiple Lines' ,'Internet Service' ,'Online Backup' ,'Device Protection' ,'Tech Support' ,'Streaming TV' ,'Streaming Movies']

    hor_space = 0.02
    ver_space = 0.02

    fig = go.FigureWidget(make_subplots(rows=4,
                                        cols=4,
                                        specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}],
                                               [{'type':'xy'}, {'type':'xy'}, {'type':'xy'}, {'type':'xy'}],
                                               [{'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}],
                                               [{'type':'xy'}, {'type':'xy'}, {'type':'xy'}, {'type':'xy'}]
                                               ],
                                        horizontal_spacing=hor_space, # in range 0 to 1/(cols-1)
                                        vertical_spacing=ver_space # in range 0 to 1/(rows-1)
                                        )
                          )

    row, col = 1, 0
    for i, (title, data_col) in enumerate(zip(titles, data_cols)):
        row, col = divmod(i, 4)
        row = row * 2

        # Get value counts for pie chart
        value_counts = df[data_col].value_counts()
        # Create pie chart trace and add to subplot
        pie_chart = go.Pie(labels=value_counts.index, values=value_counts.to_numpy(), name=title, title=title)
        fig.add_trace(pie_chart, row=row+1, col=col+1)

        # get churn rates
        churn_counts = df.groupby([data_col, 'Churn'])['Churn'].count().unstack()
        # Create stacked bar charts
        t1 = go.Bar(name='Churn (yes)', x=churn_counts['Yes'].index, y=churn_counts['Yes'])
        t2 = go.Bar(name='Churn (no)', x=churn_counts['No'].index, y=churn_counts['No'], marker_color='indianred')
        fig.add_trace(t1, row=row+2, col=col+1)
        fig.add_trace(t2, row=row+2, col=col+1)


    fig.update_layout(title="Distribution of Customer Services",
                      barmode='stack',
                      showlegend=False,
                       margin={"l":25,
                               "r":25,
                               "t":25,
                               "b":25}
                      )
    fig.show()


if __name__ == "__main__":
    main()