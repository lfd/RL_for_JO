SELECT * FROM comp_cast_type AS cct1, comp_cast_type AS cct2, movie_keyword AS mk, complete_cast AS cc, movie_info AS mi WHERE cct1.kind = 'cast' AND cct2.kind = 'complete+verified' AND mi.info IN ('Horror', 'Action', 'Sci-Fi', 'Thriller', 'Crime', 'War') AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;