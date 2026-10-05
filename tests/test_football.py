import io

from football import canon, read_fd_csv


def test_canon_maps_football_data_names():
    assert canon(" Man United ") == "Man Utd"
    assert canon("Tottenham") == "Spurs"
    assert canon("Arsenal") == "Arsenal"


def test_read_fd_csv_keeps_rows_with_trailing_fields():
    csv = "HomeTeam,AwayTeam,FTHG,FTAG\nArsenal,Chelsea,1,0\nLiverpool,Everton,2,2,,\n"
    df = read_fd_csv(io.StringIO(csv))
    assert len(df) == 2
    assert list(df.columns) == ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
