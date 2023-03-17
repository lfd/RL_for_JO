SELECT * FROM comp_cast_type AS cct1, link_type AS lt, movie_link AS ml, complete_cast AS cc, movie_companies AS mc WHERE cct1.kind = 'cast' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;