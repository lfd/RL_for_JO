SELECT * FROM movie_info_idx AS mi_idx1, company_name AS cn1, kind_type AS kt2, link_type AS lt, movie_link AS ml, title AS t2, movie_companies AS mc1 WHERE cn1.country_code != '[us]' AND kt2.kind IN ('tv series', 'episode') AND lt.link IN ('sequel', 'follows', 'followed by') AND t2.production_year BETWEEN 2000 AND 2010 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND t2.id = ml.linked_movie_id AND ml.linked_movie_id = t2.id AND cn1.id = mc1.company_id AND mc1.company_id = cn1.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id AND mi_idx1.movie_id = mc1.movie_id AND mc1.movie_id = mi_idx1.movie_id AND kt2.id = t2.kind_id AND t2.kind_id = kt2.id;