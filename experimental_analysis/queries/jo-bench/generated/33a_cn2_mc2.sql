SELECT * FROM company_name AS cn2, movie_companies AS mc2 WHERE cn2.id = mc2.company_id AND mc2.company_id = cn2.id;