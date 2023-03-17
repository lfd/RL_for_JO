SELECT * FROM link_type AS lt, movie_link AS ml, movie_companies AS mc2, company_name AS cn2 WHERE lt.link IN ('sequel', 'follows', 'followed by') AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND cn2.id = mc2.company_id AND mc2.company_id = cn2.id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id;