from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
from app.core.config import settings
from app.core.validation import validate_csv_source, validate_artifacts_integrity

class CSVTeamService:
    """Service for managing team data from CSV files and enriched information."""
    
    def __init__(self):
        self.artifacts_dir = Path(settings.ARTIFACTS_DIR)
        self.matches_file = self.artifacts_dir / settings.MATCHES_CSV_FILE
        
        # Enhanced team data with founding dates, stadiums, and cities
        self.team_enrichment_data = {
            "real madrid": {
                "official_name": "Real Madrid CF",
                "founded": 1902,
                "stadium": "Santiago Bernabéu",
                "city": "Madrid",
                "capacity": 78297,
                "nickname": "Los Blancos"
            },
            "barcelona": {
                "official_name": "FC Barcelona",
                "founded": 1899,
                "stadium": "Olímpic Lluís Companys",
                "city": "Barcelona",
                "capacity": 55926,
                "nickname": "Barça"
            },
            "ath madrid": {
                "official_name": "Atlético Madrid",
                "founded": 1903,
                "stadium": "Metropolitano",
                "city": "Madrid",
                "capacity": 70460,
                "nickname": "Rojiblancos"
            },
            "athletic club": {
                "official_name": "Athletic Club",
                "founded": 1898,
                "stadium": "San Mamés",
                "city": "Bilbao",
                "capacity": 53289,
                "nickname": "Los Leones"
            },
            "valencia": {
                "official_name": "Valencia CF",
                "founded": 1919,
                "stadium": "Mestalla",
                "city": "Valencia",
                "capacity": 49430,
                "nickname": "Los Che"
            },
            "sevilla": {
                "official_name": "Sevilla FC",
                "founded": 1890,
                "stadium": "Ramón Sánchez-Pizjuán",
                "city": "Sevilla",
                "capacity": 43883,
                "nickname": "Sevillistas"
            },
            "real sociedad": {
                "official_name": "Real Sociedad",
                "founded": 1909,
                "stadium": "Anoeta",
                "city": "San Sebastián",
                "capacity": 39313,
                "nickname": "Txuri-urdin"
            },
            "real betis": {
                "official_name": "Real Betis",
                "founded": 1907,
                "stadium": "Benito Villamarín",
                "city": "Sevilla",
                "capacity": 60721,
                "nickname": "Verdiblancos"
            },
            "villarreal": {
                "official_name": "Villarreal CF",
                "founded": 1923,
                "stadium": "La Cerámica",
                "city": "Villarreal",
                "capacity": 23008,
                "nickname": "Yellow Submarine"
            },
            "espanyol": {
                "official_name": "RCD Espanyol",
                "founded": 1900,
                "stadium": "RCDE Stadium",
                "city": "Cornellà de Llobregat",
                "capacity": 42260,
                "nickname": "Periquitos"
            },
            "getafe": {
                "official_name": "Getafe CF",
                "founded": 1946,
                "stadium": "Coliseum",
                "city": "Getafe",
                "capacity": 16500,
                "nickname": "Azulones"
            },
            "celta vigo": {
                "official_name": "RC Celta de Vigo",
                "founded": 1923,
                "stadium": "Balaídos",
                "city": "Vigo",
                "capacity": 24870,
                "nickname": "Celestes"
            },
            "osasuna": {
                "official_name": "CA Osasuna",
                "founded": 1920,
                "stadium": "El Sadar",
                "city": "Pamplona",
                "capacity": 23576,
                "nickname": "Rojillos"
            },
            "rayo vallecano": {
                "official_name": "Rayo Vallecano",
                "founded": 1924,
                "stadium": "Vallecas",
                "city": "Madrid",
                "capacity": 14708,
                "nickname": "Franjirrojos"
            },
            "mallorca": {
                "official_name": "RCD Mallorca",
                "founded": 1916,
                "stadium": "Mallorca Son Moix",
                "city": "Palma",
                "capacity": 23142,
                "nickname": "Bermellones"
            },
            "las palmas": {
                "official_name": "UD Las Palmas",
                "founded": 1949,
                "stadium": "Gran Canaria",
                "city": "Las Palmas",
                "capacity": 32392,
                "nickname": "Amarillos"
            },
            "girona": {
                "official_name": "Girona FC",
                "founded": 1930,
                "stadium": "Montilivi",
                "city": "Girona",
                "capacity": 14624,
                "nickname": "Gironins"
            },
            "alaves": {
                "official_name": "Deportivo Alavés",
                "founded": 1921,
                "stadium": "Mendizorrotza",
                "city": "Vitoria-Gasteiz",
                "capacity": 19840,
                "nickname": "Babazorros"
            },
            "leganes": {
                "official_name": "CD Leganés",
                "founded": 1928,
                "stadium": "Municipal de Butarque",
                "city": "Leganés",
                "capacity": 13089,
                "nickname": "Pepineros"
            },
            "valladolid": {
                "official_name": "Real Valladolid",
                "founded": 1928,
                "stadium": "José Zorrilla",
                "city": "Valladolid",
                "capacity": 27618,
                "nickname": "Pucelanos"
            },
            # Historical teams that may appear in older data
            "albacete": {
                "official_name": "Albacete Balompié",
                "founded": 1940,
                "stadium": "Carlos Belmonte",
                "city": "Albacete",
                "capacity": 17524,
                "nickname": "Quesos Mecánicos"
            },
            "almeria": {
                "official_name": "UD Almería",
                "founded": 1989,
                "stadium": "Power Horse Stadium",
                "city": "Almería",
                "capacity": 15274,
                "nickname": "Rojiblancos"
            },
            "cadiz": {
                "official_name": "Cádiz CF",
                "founded": 1910,
                "stadium": "Nuevo Mirandilla",
                "city": "Cádiz",
                "capacity": 20724,
                "nickname": "Submarino Amarillo"
            },
            "deportivo": {
                "official_name": "RC Deportivo",
                "founded": 1906,
                "stadium": "Riazor",
                "city": "A Coruña",
                "capacity": 32660,
                "nickname": "Depor"
            },
            "eibar": {
                "official_name": "SD Eibar",
                "founded": 1940,
                "stadium": "Ipurua",
                "city": "Eibar",
                "capacity": 8164,
                "nickname": "Armeros"
            },
            "granada": {
                "official_name": "Granada CF",
                "founded": 1931,
                "stadium": "Nuevo Los Cármenes",
                "city": "Granada",
                "capacity": 19336,
                "nickname": "Nazaríes"
            },
            "huesca": {
                "official_name": "SD Huesca",
                "founded": 1960,
                "stadium": "El Alcoraz",
                "city": "Huesca",
                "capacity": 7638,
                "nickname": "Azulgranas"
            },
            "levante": {
                "official_name": "Levante UD",
                "founded": 1909,
                "stadium": "Ciutat de València",
                "city": "Valencia",
                "capacity": 26354,
                "nickname": "Granotas"
            },
            "malaga": {
                "official_name": "Málaga CF",
                "founded": 1904,
                "stadium": "La Rosaleda",
                "city": "Málaga",
                "capacity": 30044,
                "nickname": "Boquerones"
            },
            "numancia": {
                "official_name": "CD Numancia",
                "founded": 1945,
                "stadium": "Los Pajaritos",
                "city": "Soria",
                "capacity": 8703,
                "nickname": "Numantinos"
            },
            "racing santander": {
                "official_name": "Racing de Santander",
                "founded": 1913,
                "stadium": "El Sardinero",
                "city": "Santander",
                "capacity": 22222,
                "nickname": "Racinguistas"
            },
            "real zaragoza": {
                "official_name": "Real Zaragoza",
                "founded": 1932,
                "stadium": "La Romareda",
                "city": "Zaragoza",
                "capacity": 33608,
                "nickname": "Maños"
            },
            "recreativo": {
                "official_name": "Recreativo de Huelva",
                "founded": 1889,
                "stadium": "Nuevo Colombino",
                "city": "Huelva",
                "capacity": 21670,
                "nickname": "Decano"
            },
            "sporting gijon": {
                "official_name": "Real Sporting de Gijón",
                "founded": 1905,
                "stadium": "El Molinón",
                "city": "Gijón",
                "capacity": 30000,
                "nickname": "Rojiblancos"
            },
            "tenerife": {
                "official_name": "CD Tenerife",
                "founded": 1922,
                "stadium": "Heliodoro Rodríguez López",
                "city": "Santa Cruz de Tenerife",
                "capacity": 22824,
                "nickname": "Chicharreros"
            },
            "xerez": {
                "official_name": "Xerez CD",
                "founded": 1947,
                "stadium": "Chapín",
                "city": "Jerez de la Frontera",
                "capacity": 20523,
                "nickname": "Azulinos"
            },
            "elche": {
                "official_name": "Elche CF",
                "founded": 1923,
                "stadium": "Martínez Valero",
                "city": "Elche",
                "capacity": 31388,
                "nickname": "Franjiverdes"
            },
            "oviedo": {
                "official_name": "Real Oviedo",
                "founded": 1926,
                "stadium": "Carlos Tartiere",
                "city": "Oviedo",
                "capacity": 30500,
                "nickname": "Carbayones"
            }
        }
    
    def get_teams_from_csv(self) -> List[str]:
        """Extract unique team names from the matches CSV file."""
        try:
            # Validate data source
            validate_csv_source(str(self.matches_file), "artifacts")
            validate_artifacts_integrity()
            
            # Read CSV file
            df = pd.read_csv(self.matches_file)
            
            # Extract unique teams from both HomeTeam and AwayTeam columns
            home_teams = set(df['HomeTeam'].str.lower().unique())
            away_teams = set(df['AwayTeam'].str.lower().unique())
            all_teams = sorted(home_teams.union(away_teams))
            
            return all_teams
            
        except Exception as e:
            raise Exception(f"Error reading teams from CSV: {str(e)}")
    
    def get_enriched_team_data(self, team_name: str) -> Dict:
        """Get enriched team data including founding date, stadium, and city."""
        team_key = team_name.lower().strip()
        
        base_data = {
            "name": team_name,
            "csv_name": team_name,
            "official_name": team_name.title(),
            "founded": None,
            "stadium": "Unknown",
            "city": "Unknown",
            "capacity": None,
            "nickname": "Unknown",
            "data_source": "csv_only"
        }
        
        if team_key in self.team_enrichment_data:
            enriched_data = self.team_enrichment_data[team_key]
            base_data.update({
                "official_name": enriched_data["official_name"],
                "founded": enriched_data["founded"],
                "stadium": enriched_data["stadium"],
                "city": enriched_data["city"],
                "capacity": enriched_data["capacity"],
                "nickname": enriched_data["nickname"],
                "data_source": "enriched"
            })
        
        return base_data
    
    def get_all_teams_data(self) -> List[Dict]:
        """Get all teams data with enriched information."""
        teams = self.get_teams_from_csv()
        teams_data = []
        
        for team in teams:
            team_data = self.get_enriched_team_data(team)
            teams_data.append(team_data)
        
        return teams_data
    
    def get_team_statistics(self, team_name: str) -> Dict:
        """Get basic statistics for a specific team from CSV data."""
        try:
            df = pd.read_csv(self.matches_file)
            
            # Filter matches for the specific team
            team_matches = df[
                (df['HomeTeam'].str.lower() == team_name.lower()) |
                (df['AwayTeam'].str.lower() == team_name.lower())
            ]
            
            total_matches = len(team_matches)
            home_matches = len(team_matches[team_matches['HomeTeam'].str.lower() == team_name.lower()])
            away_matches = len(team_matches[team_matches['AwayTeam'].str.lower() == team_name.lower()])
            
            return {
                "total_matches": total_matches,
                "home_matches": home_matches,
                "away_matches": away_matches,
                "data_available": total_matches > 0
            }
            
        except Exception as e:
            return {
                "total_matches": 0,
                "home_matches": 0,
                "away_matches": 0,
                "data_available": False,
                "error": str(e)
            }
    
    def get_teams_summary(self) -> Dict:
        """Get summary of all teams data."""
        teams_data = self.get_all_teams_data()
        
        enriched_count = len([t for t in teams_data if t["data_source"] == "enriched"])
        csv_only_count = len([t for t in teams_data if t["data_source"] == "csv_only"])
        
        return {
            "total_teams": len(teams_data),
            "enriched_teams": enriched_count,
            "csv_only_teams": csv_only_count,
            "enrichment_coverage": f"{(enriched_count / len(teams_data) * 100):.1f}%" if teams_data else "0%",
            "data_source": "artifacts",
            "csv_file": settings.MATCHES_CSV_FILE
        }