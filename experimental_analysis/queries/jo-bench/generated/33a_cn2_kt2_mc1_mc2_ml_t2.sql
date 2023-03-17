SELECT * FROM movie_companies AS mc1, movie_link AS ml, movie_companies AS mc2, title AS t2, kind_type AS kt2, company_name AS cn2 WHERE kt2.kind IN ('tv series') AND t2.production_year BETWEEN 2005 AND 2008 AND t2.id = ml.linked_movie_id AND ml.linked_movie_id = t2.id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id AND kt2.id = t2.kind_id AND t2.kind_id = kt2.id AND cn2.id = mc2.company_id AND mc2.company_id = cn2.id AND t2.id = mc2.movie_id AND mc2.movie_id = t2.id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id;