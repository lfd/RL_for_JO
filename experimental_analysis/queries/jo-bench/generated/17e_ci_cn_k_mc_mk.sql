SELECT * FROM keyword AS k, movie_keyword AS mk, cast_info AS ci, movie_companies AS mc, company_name AS cn WHERE cn.country_code = '[us]' AND k.keyword = 'character-name-in-title' AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;