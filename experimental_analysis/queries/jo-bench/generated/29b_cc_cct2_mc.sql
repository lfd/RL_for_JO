SELECT * FROM complete_cast AS cc, comp_cast_type AS cct2, movie_companies AS mc WHERE cct2.kind = 'complete+verified' AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;