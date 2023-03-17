SELECT * FROM movie_companies AS mc1, movie_link AS ml, movie_companies AS mc2, company_name AS cn2, company_name AS cn1 WHERE cn1.country_code = '[nl]' AND cn1.id = mc1.company_id AND mc1.company_id = cn1.id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id AND cn2.id = mc2.company_id AND mc2.company_id = cn2.id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id;