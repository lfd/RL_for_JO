SELECT * FROM movie_companies AS mc1, movie_companies AS mc2, link_type AS lt, movie_link AS ml, company_name AS cn2, title AS t2, kind_type AS kt2 WHERE kt2.kind IN ('tv series', 'episode') AND lt.link IN ('sequel', 'follows', 'followed by') AND t2.production_year BETWEEN 2000 AND 2010 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND t2.id = ml.linked_movie_id AND ml.linked_movie_id = t2.id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id AND kt2.id = t2.kind_id AND t2.kind_id = kt2.id AND cn2.id = mc2.company_id AND mc2.company_id = cn2.id AND t2.id = mc2.movie_id AND mc2.movie_id = t2.id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id;