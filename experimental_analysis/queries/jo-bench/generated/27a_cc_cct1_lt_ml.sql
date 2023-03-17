SELECT * FROM comp_cast_type AS cct1, complete_cast AS cc, link_type AS lt, movie_link AS ml WHERE cct1.kind IN ('cast', 'crew') AND lt.link LIKE '%follow%' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id;