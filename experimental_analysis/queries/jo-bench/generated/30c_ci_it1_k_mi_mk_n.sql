SELECT * FROM movie_keyword AS mk, info_type AS it1, movie_info AS mi, cast_info AS ci, keyword AS k, name AS n WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info = 'genres' AND k.keyword IN ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND mi.info IN ('Horror', 'Action', 'Sci-Fi', 'Thriller', 'Crime', 'War') AND n.gender = 'm' AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;